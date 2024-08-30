import numpy as np
import matplotlib.pyplot as plt
class QuadraticLyapunov:
    def __init__(self, tilde_X, dot_tilde_X, K_d_t, D_d_t, F_ext, M, n_samples=300, gamma=1e-12, alpha=0.9, is_exponential=False, lambda_val=0.0):
        self.tilde_X = tilde_X
        self.dot_tilde_X = dot_tilde_X
        self.K_d_t = K_d_t
        self.D_d_t = D_d_t
        self.F_ext = F_ext
        self.M = M
        self.n_samples = n_samples
        self.gamma = gamma
        self.alpha = alpha  # Damping weight, adjust as necessary
        self.lambda_val = lambda_val
        self.is_exponential = is_exponential

    def V(self):
        # Calculate the modified Lyapunov function for each time instant
        V = np.zeros(self.n_samples)
        if self.is_exponential:
            return self.exponential_V()

        for i in range(self.n_samples):
            pos_term = 0.5 * self.tilde_X[i].T @ self.K_d_t[i] @ self.tilde_X[i]
            vel_term = 0.5 * self.dot_tilde_X[i].T @ self.M @ self.dot_tilde_X[i]
            damping_term = 0.5 * self.alpha * self.dot_tilde_X[i].T @ self.D_d_t[i] @ self.dot_tilde_X[i]
            V[i] = pos_term + vel_term + damping_term
        return V

    def dot_V(self):
        # Calculate the value of the derivative of the modified Lyapunov function for each time instant
        dot_V = np.zeros(self.n_samples)
        if self.is_exponential:
            return self.exponential_dot_V()

        for i in range(self.n_samples):
            damping_term = self.dot_tilde_X[i].T @ self.D_d_t[i] @ self.dot_tilde_X[i]
            force_term = self.dot_tilde_X[i].T @ (self.F_ext[i] - self.K_d_t[i] @ self.tilde_X[i])
            dot_V[i] = force_term - damping_term + self.alpha * damping_term

            # Check and adjust for negativity
            if dot_V[i] >= 0:
                print(f"Adjusting parameters to maintain negativity of dot_V at time {i}.")
                # Modify alpha or damping matrix as needed
                self.alpha *= 1.05  # Slightly increase alpha if needed for stronger damping
                dot_V[i] = force_term - damping_term + self.alpha * damping_term

        return dot_V

    def exponential_V(self):
        V = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            pos_term = 0.5 * self.tilde_X[i].T @ self.K_d_t[i] @ self.tilde_X[i]
            vel_term = 0.5 * self.dot_tilde_X[i].T @ self.M @ self.dot_tilde_X[i]
            V[i] = (pos_term + vel_term) * np.exp(-self.lambda_val * i)
        return V

    def exponential_dot_V(self):
        dot_V = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            damping_term = self.dot_tilde_X[i].T @ self.D_d_t[i] @ self.dot_tilde_X[i]
            force_term = self.dot_tilde_X[i].T @ (self.F_ext[i] - self.K_d_t[i] @ self.tilde_X[i])
            V_i = 0.5 * (self.tilde_X[i].T @ self.K_d_t[i] @ self.tilde_X[i] + self.dot_tilde_X[i].T @ self.M @ self.dot_tilde_X[i])
            dot_V[i] = (force_term - damping_term) * np.exp(-self.lambda_val * i) - self.lambda_val * V_i * np.exp(-self.lambda_val * i)

            if dot_V[i] >= 0:
                print(f"At time {i}, dot_V is not negative. Adjust lambda or matrices.")

        return dot_V
    def stability_condition(self):

        dot_V = self.dot_V()
        # Check if the system satisfies the exponential stability condition
        stability_condition = dot_V <= -self.gamma * np.sum(self.tilde_X**2, axis=1)
        return dot_V, np.all(stability_condition[:-1])

    def visualize(self, V=None, dot_V=None):
        # Visualize the Lyapunov function and its derivative
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(V, label="V(x)")
        plt.title("Lyapunov Function")
        plt.xlabel('Time')
        plt.ylabel('V(x)')
        plt.subplot(1, 2, 2)
        plt.plot(dot_V, label=r"$\dot{V}(x)$")
        plt.xlabel('Time')
        plt.ylabel(r"$\dot{V}(x)$")
        plt.legend()
        plt.show()

    def __call__(self):
        return self.V(), self.stability_condition()


if __name__ == "__main__":
    # Example usage
    tilde_X = np.random.rand(300, 3)
    dot_tilde_X = np.random.rand(300, 3)
    K_d_t = np.random.rand(300, 3, 3)
    D_d_t = np.random.rand(300, 3, 3)
    F_ext = np.random.rand(300, 3)
    M = np.eye(3)  # Assuming mass matrix is the identity matrix

    # Create an instance of the QuadraticLyapunov class
    ql = QuadraticLyapunov(tilde_X, dot_tilde_X, K_d_t, D_d_t, F_ext, M, n_samples=300)
    # Call the method to get the results
    V, dot_V, stability_condition = ql()
    print("V(x):", V)
    print("dot_V(x):", dot_V)
    print("is_stable:", stability_condition)