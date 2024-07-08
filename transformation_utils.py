import numpy as np
from numpy import cos, sin
def transform_coordinates(th_x_deg=0, th_y_deg=0, th_z_deg=0):
    th_x, th_y, th_z = np.deg2rad(th_x_deg), np.deg2rad(th_y_deg), np.deg2rad(th_z_deg)
    R_x = np.array([
        [1, 0, 0],
        [0, cos(th_x), -sin(th_x)],
        [0, sin(th_x), cos(th_x)]
    ])
    R_y = np.array([
        [cos(th_y), 0, sin(th_y)],
        [0, 1, 0],
        [-sin(th_y), 0, cos(th_y)]
    ])
    R_z = np.array([
        [cos(th_z),-sin(th_z), 0],
        [sin(th_z),cos(th_z), 0],
        [0, 0, 1]
    ])

    # Combine the rotations
    R = np.dot(R_x,R_y)

    return R

def transform_xyz_to_zxy(X):
    # Ensure the input is a NumPy array of length 3
    if not isinstance(X, np.ndarray) or X.shape != (3,):
        raise ValueError("Input must be a NumPy array of length 3.")

    # Rearrange (x, y, z) to (z, x, y)
    return np.array([X[2], X[0], X[1]])