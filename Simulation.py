from enum import Enum
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy import cos, sin
from transformation_utils import transform_coordinates
from vic_controller_with_tank_energy_inside import VICController
# from vic_controller_with_tank_energy_with_convergence_measures import VICController
from sklearn.metrics import root_mean_squared_error as rmse
from time import time

from visualization_utils import VisualizerOnline, VisualizerOffline, VisualizerOfflineSameRanges
EXP_DATA_PATH = './experimental_data'
# Load the data
class MaterialMeta(Enum):
    PVC = 'pvc'
    penoplex = 'peno'
    cork =  'crck'

class MaterialData:
    PVC = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.PVC.value}.npy')
    penoplex = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.penoplex.value}.npy')
    cork = np.load(f'./predicted_pose_twist_wrench_{MaterialMeta.cork.value}.npy')


data_gmr = MaterialData.cork
MATERIAL_NAME = MaterialMeta.cork.name
title = f"Visualization of Dynamics during cutting {MATERIAL_NAME}"

# Optimizer configs
use_k_min = False
use_k_max = False
use_specified_k = False
k_specified = [500,500,500] # To test using constant stiffness case
print(data_gmr.shape)

# Define indexes for clarity
pos_idx = np.s_[0:3]
vel_idx = np.s_[12:15]
force_idx = np.s_[6:9]
pos = data_gmr[:, pos_idx]
pos = pos[:,[1,0,2]] # Transform ZXY to ZYX
vel = data_gmr[:, vel_idx]
force = data_gmr[:, force_idx]

F_max = np.max(force, axis=0)
F_min = np.min(force, axis=0)

class MotionPlanner:
    def __init__(self, pos: np.ndarray, twist: np.ndarray, wrench: np.ndarray) -> None:
        self.P = pos
        self.V = twist
        self.W = wrench

    def go_to_next(self):
        """ Generator that yields position, velocity, and force at each step. """
        for x, v, f in zip(self.P, self.V, self.W):
            yield x, v, f


if __name__ == "__main__":
    # tcp_rot = transform_coordinates(th_x_deg=90, th_y_deg=90)
    plot_desired_pos = False
    if plot_desired_pos:
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot(pos[:,2],pos[:,1], pos[:,0], color='k')
        ax.set_title(label=f"Desired trajextory expressed in robot tcp frame - {MATERIAL_NAME} material")
        ax.set_xlabel(r'Z_{tcp} [m]')
        ax.set_ylabel(r'Y_{tcp}[m]')
        ax.set_zlabel(r'X_{tcp}[m]')
        plt.tight_layout()
        ax.view_init(elev=65, azim=-30)
        plt.show()
    controller = VICController(F_min=F_min, F_max=F_max)
    planner = MotionPlanner(pos=pos, twist=vel, wrench=force)
    # visualizer = Visualizer(xlim=(min(pos[:,-1]), max(pos[:,-1])), ylim=(0,1))

    visualizer = VisualizerOfflineSameRanges(title=title)

    for x, x_dot, F_d in planner.go_to_next():
        x_tilde = np.abs(pos[-1,:] - x)
        x_tilde_dot = np.abs(vel[-1,:] - x_dot)
        start_time = time()
        if not use_k_min and not use_k_max and not use_specified_k:
            kd_opt, dd_opt = controller.optimize(x_tilde, x_tilde_dot, F_d)
        else:
            if use_k_min:
                kd_opt = controller.k_min
                dd_opt = np.array([0.7, 0.7, 0.7])
            elif use_k_max:
                kd_opt = controller.k_max
                dd_opt = np.array([0.7, 0.7, 0.7])
            elif use_specified_k:
                kd_opt = k_specified
                dd_opt = np.array([0.7, 0.7, 0.7])
            controller.K_d = kd_opt
            dd_opt = dd_opt

        F_actual = controller.calculate_force(x_tilde, x_tilde_dot)
        elapsed_time = time() - start_time
        print(f"Elapsed time for this step: {elapsed_time}s")
        print(f"\t k_d = {kd_opt}, dd = {dd_opt}, F_act = {F_actual}")
        print("Next Position:", x_tilde, "Next Force:", F_d)
        # print("Total energy: "+str(controller.E_tot))
        visualizer.collect_data(x=x_tilde,Fd=F_d, Fext=F_actual,k=kd_opt, d=dd_opt)
    visualizer.show()


    print("Completed all steps.")
    plt.plot(controller.E_tot)
    plt.xlabel('Step')
    plt.ylabel(r'Tank storage $T(x_t)$: '+ MATERIAL_NAME)
    plt.show()


