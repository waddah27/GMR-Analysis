import numpy as np
import matplotlib.pyplot as plt
data_gmr = np.load('./predicted_pose_twist_wrench_peno.npy')
print(data_gmr.shape)
X, Y, Z = data_gmr[:,0], data_gmr[:,1], data_gmr[:,2] 
Fx, Fy, Fz = data_gmr[:,6], data_gmr[:,7], data_gmr[:,8]

ax1 = plt.subplot(211)
plt.plot(X,Z)
ax1.set_xlabel('X')
ax1.set_ylabel('Z')

ax2 = plt.subplot(212)
plt.plot(Fz)
ax2.set_xlabel('time')
ax2.set_ylabel('Fz')


plt.show()



ax1 = plt.subplot(311)
plt.plot(Fx)
ax1.set_xlabel('time')
ax1.set_ylabel('Fx')

ax2 = plt.subplot(312)
plt.plot(Fy)
ax2.set_xlabel('time')
ax2.set_ylabel('Fy')

ax3 = plt.subplot(313)
plt.plot(Fz)
ax3.set_xlabel('time')
ax3.set_ylabel('Fz')

plt.show()
