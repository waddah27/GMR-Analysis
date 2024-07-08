from typing import Optional
import numpy as np
from numpy import ndarray
from scipy.optimize import minimize

class VICController:
    def __init__(self, F_min: Optional[ndarray] = None, F_max: Optional[ndarray] = None):
        self.Xi_scaler = 5000
        self.k_min = np.array([10, 10, 10])
        self.k_max = np.array([5000, 5000, 5000])
        self.K_d = self.k_min
        self.D_min = np.array([0, 0, 0])
        self.D_max = np.array([self.Xi_scaler, self.Xi_scaler, self.Xi_scaler])
        self.D_d = np.array([0.7, 0.7, 0.7]) * self.Xi_scaler
        self.f_min = -70 if F_min is None else F_min
        self.f_max = 70 if F_max is None else F_max
        self.epsilon = 0.675 #0.4 0.9
        self.scaler = 1
        self.Q = np.eye(3)
        self.R = np.eye(3) * 1e-9
        self.delta_t = 0.1
        self.K_d_list = []
        self.E_t = 0
        self.E_tot = []
        self.x_t = np.array([self.epsilon, self.epsilon, self.epsilon])
        # self.x_t = np.array([0.9, 0.9, 0.9])
        self.T_max = 3000 ##TODO: To be used as extra block

        self.convergence_threshold = 1e-6  # Convergence threshold for objective function
        self.max_iterations = 100  # Maximum number of optimization iterations
        self.error_threshold = 1e-3  # Error threshold for variables

    def objective(self, params, x_tilde, x_tilde_dot, F_d):
        k_d = np.diag(params[:3])
        xi_d = params[3:6]
        xi_d = xi_d/self.Xi_scaler
        d_d = self.calculate_damping(xi_d, k_d)
        F_ext = np.dot(k_d, x_tilde) + np.dot(d_d, x_tilde_dot)

        norm_F = np.dot((F_ext - F_d).T, np.dot(self.Q, (F_ext - F_d)))
        norm_k = np.dot((np.diag(k_d) - self.k_min).T, np.dot(self.R, (np.diag(k_d) - self.k_min)))
        smoothness_penalty = np.sum(np.diff(xi_d)**2)
        force_penalty = np.sum(np.maximum(0, F_ext - self.f_max) ** 2) + np.sum(np.maximum(0, self.f_min - F_ext) ** 2)

        # Energy constraint penalty
        passivity_penality, energy_penalty, self.E_t = self.tank_energy_penalty(params, x_tilde, x_tilde_dot, F_d)
        # self.E_tot.append(self.E_t)

        return norm_F + norm_k + force_penalty + energy_penalty + passivity_penality #+ smoothness_penalty

    def calculate_damping(self, xi_d, k_d):
        sqrt_k_d = np.sqrt(k_d)
        return 2 * np.diag(xi_d) * sqrt_k_d

    def tank_energy_penalty(self, params, x_tilde, x_tilde_dot, F_d):
        k_d = np.diag(params[:3])
        xi_d = params[3:6]
        xi_d = xi_d/self.Xi_scaler
        d_d = self.calculate_damping(xi_d, k_d)
        k_v = self.K_d - self.k_min
        # x_t_dot = self.tank_dynamics(self.x_t, x_tilde_dot, k_v, d_d)
        x_t_dot = self.T_dot(self.x_t,x_tilde, x_tilde_dot, k_v, d_d)/self.x_t
        x_t_next = self.x_t + x_t_dot * self.delta_t
        E = self.T(x_t_next)
        self.x_t = x_t_next

        passivity_penalty = max(0, self.epsilon - E) ** 2  # Penalty for energy constraint violation T(x_t)> self.epsilon
        energy_penality = max(0,E - self.T_max)
        return passivity_penalty, energy_penality, E

    def optimize(self, x_tilde, x_tilde_dot, F_d):
        xi_initial = 0.7 * self.Xi_scaler
        initial_guess = [200, 200, 200, xi_initial, xi_initial, xi_initial]
        bounds = [(self.k_min[i], self.k_max[i]) for i in range(3)] + [(self.D_min[i], self.D_max[i]) for i in range(3)]

        result = minimize(
            self.objective,
            initial_guess,
            args=(x_tilde, x_tilde_dot, F_d),
            bounds=bounds,
            method='L-BFGS-B'
        )

        self.K_d = result.x[:3]
        self.D_d = result.x[3:6]/self.Xi_scaler
        self.E_tot.append(self.E_t)

        return self.K_d, self.D_d

    def tank_dynamics(self, x_t, x_tilde_dot, K_v, D_d):
        return -K_v * np.sum(x_tilde_dot) - np.sum(D_d * x_tilde_dot)

    def T_dot(self, x_t,x_tilde, x_tilde_dot, K_v, D_d):
        sigma = 1 if self.T(x_t) > self.epsilon else 0
        w = (-K_v * np.sum(x_tilde)/x_t) if self.T(x_t) > self.epsilon else 0
        return (sigma * np.dot(x_tilde_dot.T, np.dot(D_d, x_tilde_dot)))/x_t - np.dot(w, x_tilde_dot)

    def T(self, x_t):
        return 0.5 * np.sum(x_t ** 2)

    def calculate_force(self, x_tilde, x_tilde_dot):
        kd = np.diag(self.K_d)
        dd = np.diag(self.D_d)
        return np.dot(kd, x_tilde) + np.dot(dd, x_tilde_dot)
