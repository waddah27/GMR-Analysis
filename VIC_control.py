#!/bin/env python3
import time
from typing import Optional, Union

from numpy import ndarray
import rospy
import tf
import copy
import threading
import numpy as np
import pandas as pd
from ros_numpy import geometry
from std_msgs.msg import Int32
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Vector3, Point, Quaternion, TwistStamped
from iiwa_msgs.srv import (
    ConfigureControlMode, 
    ConfigureControlModeRequest, 
    SetSmartServoLinSpeedLimits, 
    SetSmartServoLinSpeedLimitsRequest,
    SetPTPJointSpeedLimits,
    SetPTPCartesianSpeedLimits,
    SetSmartServoJointSpeedLimits,
    SetPTPCartesianSpeedLimitsRequest,
    

)

from iiwa_msgs.msg import (
    CartesianQuantity,
    CartesianPose, 
    CartesianVelocity,
    JointVelocity,
    JointPosition, 
    JointQuantity,
    MoveToCartesianPoseAction,
    CartesianWrench,
    CartesianPlane,
    ControlMode,
    DOF
)

import tf.transformations

import actionlib
# from GMR_Analysis.vic_stiffness_optimizer import optimize_stiffness
from GMR_Analysis.vic_stiffness_optimizer_with_tank_energy import optimize_stiffness
from vic_stiffness_optimizer_scipy import OptimalStiffness
from vic_controller_with_tank_energy import VICController
import cvxpy as cp

lock = threading.Lock()

geometric_point = geometry.Point 
class ExampleImpedanceControl:

    def __init__(self):

        # configuration services
        self.conf_srv = rospy.ServiceProxy('/iiwa/configuration/ConfigureControlMode', ConfigureControlMode)
        self.conf_limits_srv = rospy.ServiceProxy('/iiwa/configuration/setSmartServoLinLimits', SetSmartServoLinSpeedLimits)
        self.p2pjs_limits_srv = rospy.ServiceProxy('/iiwa/configuration/setPTPJointLimits', SetPTPJointSpeedLimits)
        self.p2pcart_limits_srv = rospy.ServiceProxy('/iiwa/configuration/setPTPCartesianLimits', SetPTPCartesianSpeedLimits)
        self.jsvel_limits_srv = rospy.ServiceProxy('/iiwa/configuration/setSmartServoLimits', SetSmartServoJointSpeedLimits)
        

        # state readers
        rospy.Subscriber('/iiwa/state/CartesianPose', CartesianPose, self.cartesian_pose_callback)
        # rospy.Subscriber('/iiwa/state/CartesianVelocity', CartesianVelocity, self.cartesian_vel_callback)
        rospy.Subscriber('/iiwa/joint_states', JointState, self.js_callback)
        rospy.Subscriber('/iiwa/state/CartesianWrench', CartesianWrench, self.cartesian_wrench_callback)
        
        # control writers
        self.js_control_pub = rospy.Publisher('/iiwa/command/JointPosition', JointPosition, queue_size=10)
        self.cart_control_pub = rospy.Publisher('/iiwa/command/CartesianPoseLin', PoseStamped, queue_size=10)
        self.p2pcart_control_pub = rospy.Publisher('/iiwa/command/CartesianPose', PoseStamped, queue_size=10)
        

        self.cartlin_client = actionlib.SimpleActionClient('/iiwa/action/move_to_cartesian_pose', 
                                                           MoveToCartesianPoseAction)

        self.q = self.dq = self.tau_ext = None # \in R^n
        self.x = self.dx = None   # \in R^3 = [x,y,z]
        self.h = None   # current tcp quaterinon
        self.cart_wrench = self.cart_wrench_err = None # cartesian wrench
        self.tcp_force = self.tcp_torque = self.tcp_wrench = self.tcp_pose_orient = None
        self.force_error = None
        self.tank_energy = None
        self.tcp_pose = self.tcp_vel = None    # rosmsg to copy
        
        
        self.tcp_k = [np.array([10,10,10])]
        # Motion plan
        # self.start_pt = [0.24, -0.51000, 0.03828] # the object plate with FT sensor frame
        # self.end_pt = [-0.06314,self.start_pt[1], self.start_pt[2]]
        # self.pt_hieght = self.start_pt[-1] + 0.05
        # self.pt_cut_depth = 0.00344
        # self.traj_disp = np.linspace(self.start_pt[0],self.end_pt[0],10) 
        
        self.n_steps = 100
        self.start_pt = [0.28293, -0.630419, 0.03828] # the object plate with FT sensor frame
        self.end_pt = [-0.09853,-0.80060, self.start_pt[2]]
        self.pt_hieght = self.start_pt[-1] + 0.05
        self.pt_cut_depth = 0.00344
        self.x_disp = np.linspace(self.start_pt[0],self.end_pt[0],self.n_steps) 
        self.y_disp = np.linspace(self.start_pt[1],self.end_pt[1],self.n_steps)
        self.scalpel_ang = -0.6653195108602383 #-0.36791541 #0.21618
        
        ## Stifness optimization configs
        self.dt = 0.1
        # self.stiffness_optimizer = optimize_stiffness(
        #     n_dims=3,
        #     # k_min=np.array([500, 500, 500]),
        #     # k_max=np.array([500, 500,2000]),
        #     k_min=np.array([50, 50, 50]),
        #     k_max=np.array([500, 500,500]),
        #     solver=cp.OSQP,
        #     epsilon=0.3,
        # )
        self.stiffness_optimizer = OptimalStiffness(
            n_dims=3,
            k_min=np.array([50, 50, 50]),
            k_max=np.array([500, 500,500]),
        )
        self.vic_controller = VICController()
        

        # tcp_x --> - Base_fz
        

        # log file
        self.wrench_out_file = open('wrench_log.txt', 'w')
        self.pose_out_file = open('pose_log.txt', 'w')
        self.time_out_file = open('timer_log.txt', 'w')

        rospy.on_shutdown(self.shutdown)

    def shutdown(self):
        # stop a robot if the program is closed
        self.activate_p2p_in_js_mode()
        q = copy.deepcopy(self.q)
        self.move_to_pose_js(q)

    def __del__(self):
        self.wrench_out_file.close()

    def setup_lin_limits(self, linear_limits, angular_limits):
        """WARN: must be called to activate cartesion control modes"""
        lin_limtits_msg = SetSmartServoLinSpeedLimitsRequest()
        lin_limtits_msg.max_cartesian_velocity.linear = linear_limits
        lin_limtits_msg.max_cartesian_velocity.angular = angular_limits
        self.conf_limits_srv(lin_limtits_msg)
    
    def setup_p2pcart_limits(self, linear_limits=0.01, angular_limits=0.005):
        """WARN: must be called to activate cartesion control modes"""
        lin_limtits_msg = SetPTPCartesianSpeedLimitsRequest()
        lin_limtits_msg.maxCartesianVelocity = linear_limits
        lin_limtits_msg.maxOrientationVelocity = angular_limits
        self.p2pcart_limits_srv(lin_limtits_msg)

    def activate_p2p_in_js_mode(self):
        req_msg = ConfigureControlModeRequest()
        req_msg.control_mode = 0
        self.conf_srv(req_msg)

    def activate_joint_impedance_mode(self, stiffness, damping):
        req_msg = ConfigureControlModeRequest()
        req_msg.control_mode = ControlMode.JOINT_IMPEDANCE
        req_msg.joint_impedance.joint_stiffness = stiffness
        req_msg.joint_impedance.joint_damping = damping
        self.conf_srv(req_msg)

    def activate_cartision_impedance_mode(self, stiffness, damping):
        req_msg = ConfigureControlModeRequest()
        req_msg.control_mode = 2
        req_msg.cartesian_impedance.cartesian_stiffness = stiffness
        req_msg.cartesian_impedance.cartesian_damping = damping
        self.conf_srv(req_msg)
    
    def activate_force_control_mode(self, desired_force, cartesian_dof=DOF.X):
        req_msg = ConfigureControlModeRequest()
        req_msg.control_mode = 3
        req_msg.desired_force.desired_force = desired_force
        req_msg.desired_force.cartesian_dof = cartesian_dof
        req_msg.desired_force.desired_stiffness = 1000
        
        self.conf_srv(req_msg)
    
    def activate_sine_pattern_control_mode(self, amplitude=10, cartesian_dof=DOF.X, frequency=10):
        req_msg = ConfigureControlModeRequest()
        req_msg.control_mode = ControlMode.SINE_PATTERN
        req_msg.sine_pattern.amplitude = amplitude
        req_msg.sine_pattern.cartesian_dof = cartesian_dof
        req_msg.sine_pattern.frequency = frequency
        req_msg.sine_pattern.stiffness = 500
        self.conf_srv(req_msg)

    def move_to_pose_js(self, q):
        """p2p in joint space"""
        jp = JointPosition()
        jp.header.seq = 1
        jp.header.stamp = rospy.Time.now()
        jp.header.frame_id = "world"
        jp.position = JointQuantity(q[0], q[1], q[2], q[3], q[4], q[5], q[6])
        self.js_control_pub.publish(jp)


    def move_to_pose(self, point, orientation):
        """p2p in Cartesion space"""
        if self.tcp_pose:
            # print(f"Go to {point.x} {point.y} {point.z}, current tcp pos: {self.x[0]} {self.x[1]} {self.x[2]}")
            pose_stamped = copy.deepcopy(self.tcp_pose.poseStamped)
            pose_stamped.pose.position = point
            # pose_stamped.pose.orientationkd_opt = np.array([10, 10, 10]) = orientation
            self.p2pcart_control_pub.publish(pose_stamped)

        else:
            raise Exception('There is no current tcp pose')

    def move_lin(self, dx:geometric_point, h:Quaternion):
        """ 
            infinetly moveing with spetified velocity
            x is linear velocity by [x,y,z]
        """
        if self.tcp_pose:
            pose_stamped = copy.deepcopy(self.tcp_pose.poseStamped)
            pose_stamped.pose.position.x = dx.x
            pose_stamped.pose.position.y = dx.y
            pose_stamped.pose.position.z = dx.z
            pose_stamped.pose.orientation = h
            self.cart_control_pub.publish(pose_stamped)
        else:
            raise Exception('There is no current tcp pose')

    def cartesian_pose_callback(self, msg: CartesianPose):
        """state"""
        lock.acquire()
        self.tcp_pose = msg
        self.x = np.array(geometry.vector3_to_numpy(msg.poseStamped.pose.position))
        self.h = msg.poseStamped.pose.orientation
        # h_rad = np.array(tf.transformations.euler_from_quaternion(self.h))
        # self.tcp_pose_orient = np.concatenate([self.x, h_rad], axis=None)
        lock.release()
    
    def cartesian_vel_callback(self, msg: CartesianVelocity):
        """state"""
        lock.acquire()
        self.tcp_vel = msg
        # self.dx = np.array(geometry.vector3_to_numpy(msg.velocity))
        lock.release()
    

    def js_callback(self, msg: JointState):
        """state"""
        lock.acquire()
        self.q = np.array(msg.position)
        self.dq = np.array(msg.velocity)
        self.tau_ext = np.array(msg.effort)
        lock.release()
    
    def cartesian_wrench_callback(self, msg: CartesianWrench):
        lock.acquire()
        self.cart_wrench = msg.wrench
        self.cart_wrench_err = msg.inaccuracy
        self.tcp_force = np.array(geometry.vector3_to_numpy(self.cart_wrench.force))
        self.tcp_torque = np.array(geometry.vector3_to_numpy(self.cart_wrench.torque))
        self.tcp_wrench = np.concatenate([self.tcp_force, self.tcp_torque], axis=None)
        lock.release()

    def is_achived(self, error:float, tolerance:float):
        # print(np.linalg.norm(error))
        if np.linalg.norm(error) < tolerance:
            return True
        return False
    
    def log_results(self, tcp_wrench:Union[list,CartesianWrench,None]=None, 
                    tcp_pose_orient: Union[list,CartesianPose,None] = None, 
                    tcp_k:Union[list,ndarray,CartesianQuantity,None] = None,
                    tcp_d:Union[list, ndarray,CartesianQuantity,None]=None,
                    force_error:Union[list, ndarray, CartesianQuantity, None] = None,
                    time:Union[float,None]=None):
        if time is not None:
            self.time_out_file.writelines(f"{time} : {self.tcp_force} \n")
        if tcp_wrench is not None:
            print(f" tcp Force = {self.tcp_force}")
            for i in tcp_wrench:
                self.wrench_out_file.write(str(f"{i} "))
        if tcp_k is not None:
            for i in tcp_k:
                self.wrench_out_file.write(str(f"{i} "))
            # self.wrench_out_file.writelines("\n")
            # self.wrench_out_file.writelines(str(f"{self.tcp_wrench} \n"))
        if tcp_d is not None:
            for i in tcp_d:
                self.wrench_out_file.write(str(f"{i} "))
            
        if force_error is not None:
            self.wrench_out_file.write(str(f"{force_error} "))
        
        self.wrench_out_file.writelines("\n")
        
        if tcp_pose_orient is not None:
            # print(f"tcp_pos = {self.x, self.h}")
            for i in tcp_pose_orient:
                self.pose_out_file.write(str(f"{i} "))
            self.pose_out_file.writelines("\n")

    def spin(
            self, goto_init:bool = True, 
             move:bool = True, cartesian_control = True, 
             force_control = False, pos_control=False, record:bool = True, start_disp:float = 0.0):

        q0 = np.array([
            -0.74058133,  
            1.05021501, 
            -0.75677675, 
            -1.49924064,  
            0.94548076,  
            0.90629095,
            -1.76560926
        ])

        self.start_pt[1]+=start_disp # displace the trajectory on +y dir
        path = [
            [i, j, self.pt_cut_depth] for i,j in zip(self.x_disp, self.y_disp) 

        ]
        path.insert(0,[self.start_pt[0], self.start_pt[1], self.pt_hieght])
        path.append([self.end_pt[0], self.end_pt[1], self.pt_hieght])
        x_tilde = np.array([0.0, 0.0, 0.0]) ## THIS SHOULD BE THE ERR x_d - x_tcp
        # F_d = np.array([0, 0, 15]) ## THIS SHOULD BE THE DESIRED FORCE FROM GMR
        human_data = np.load('./predicted_pose_twist_wrench_peno.npy')
        F_d_gmr = human_data[:,6:9]
        kd_opt = self.tcp_k[-1]
        dd_opt = np.array([0.7, 0.7, 0.7])
               
        self.setup_lin_limits(Vector3(0.1, 0.1, 0.1), Vector3(0.1, 0.1, 0.1)) #Vector3(0.05, 0.05, 0.05)
        RT = 1000
        init = goto_init
        i = 0
        t = 0
        rate = rospy.Rate(RT) #30
        start_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            t = rospy.get_time() - start_time
            
            # print("JOINTS CONFIG: "+str(self.q))
            # print("Desired stiffness/wrench"+str(self.tcp_force))
            if init:
                print('init')
                # print(f"tcp pose = {self.x}")
                
                # go to initial configuration
                self.activate_p2p_in_js_mode()
                self.move_to_pose_js(q0)

                if self.is_achived(self.q - q0, tolerance=0.01):
                    init = False
                    

            else:
                point = np.array(path[i])
                if cartesian_control:
                    print('Cartesian')
                    x_err = np.linalg.norm(point - self.x)
                    
                    # x_tilde[0] = np.linalg.norm(point[0] - self.x[0])
                    # x_tilde[1] = np.linalg.norm(point[1] - self.x[1])
                    # x_tilde[2] = np.linalg.norm(point[2] - self.x[2])
                    x_tilde = point
                    F_d = F_d_gmr[i]
                    # x_tilde = point - self.x
                    x_tilde_dot = np.array([0.1, 0.1, 0.1])
                    print("POS ERROR -- ", x_tilde)
                    
                    try:
                        kd_opt, dd_opt = self.vic_controller.optimize(x_tilde, x_tilde_dot, F_d)
                        if kd_opt is not None:
                            self.tcp_k.append(kd_opt)
                    except Exception as E:
                        print(f"Failed! Reason:{E}")
                        # kd_opt = self.tcp_k[-1]
                        # dd_opt = np.array([0.7, 0.7, 0.7])
                    
                    # # setup cartesion stifftess
                    print("Kd_opt = "+str(kd_opt))
                    # self.activate_cartision_impedance_mode(
                    #     CartesianQuantity(kd_opt[0], kd_opt[1], kd_opt[2], 500, 500, 500), 
                    #     CartesianQuantity(0.7,0.7,0.7,0.7,0.7,0.7))
                    self.activate_cartision_impedance_mode(
                        CartesianQuantity(kd_opt[0], kd_opt[1], kd_opt[2], 500, 500, 500), 
                        # CartesianQuantity(0.7,0.7,0.7,0.7,0.7,0.7))
                        CartesianQuantity(dd_opt[0],dd_opt[1],dd_opt[2],0.7,0.7,0.7))
                
                if move:
                    print(f"GO TO POINT {i}")
                    
                    # point = np.array(path[i])
                    if self.h is not None:
                        h_rad = list(tf.transformations.euler_from_quaternion([self.h.x, self.h.y, self.h.z, self.h.w]))
                        h_rad[-1] = self.scalpel_ang
                        self.tcp_pose_orient = np.concatenate([self.x, h_rad], axis=None)
                        _val = tf.transformations.quaternion_from_euler(*h_rad)
                        h = Quaternion(_val[0], _val[1], _val[2], _val[3])
                        # print('TCP ORIENTATION IN DEGS: '+str(h_rad))
                    else:
                        h = None
                    # h = self.h
                    p = geometry.numpy_to_point(point)
                    self.move_lin(p, h)
                    
                    if self.is_achived(point - self.x, tolerance=0.005):
                        # if i > 0 and i<len(path)-1:
                        #     self.activate_force_control_mode(desired_force=self.tcp_force[-1], cartesian_dof=DOF.Z)

                        i += 1
                        if i >= len(path):
                            i = 0
                            
            if record:
                self.log_results(tcp_wrench=self.tcp_wrench,
                                 tcp_pose_orient=self.tcp_pose_orient, 
                                 tcp_k=kd_opt,
                                 tcp_d=dd_opt, 
                                 force_error=self.force_error,
                                 time=t)
                # print(self.q)
                


            rate.sleep()

if __name__ == "__main__":
    rospy.init_node('test_path_node')

    tp = ExampleImpedanceControl()
    tp.spin(
        goto_init=True, 
        move=True, 
        cartesian_control=True, 
        force_control=False, 
        pos_control=False,
        record=True,
        start_disp=0.0)