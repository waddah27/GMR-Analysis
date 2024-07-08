import matplotlib.pyplot as plt
import numpy as np

class VisualizerOnline:
    def __init__(self, title:str) -> None:
        # Creating subplots
        self.fig, self.axs = plt.subplots(2, 3, figsize=(20, 14))
        # Set a general title for the figure
        self.fig.suptitle(title, fontsize=14)
        self.positions = [r'$\tilde{X}$', r'$\tilde{Y}$', r'$\tilde{Z}$']
        self.forces = [r'$F_x$', r'$F_y$', r'$F_z$']
        self.stiffness = [r'$k^d_x$', r'$k^d_y$', r'$k^d_z$']
        self.damping = [r'$\xi^{d}_{x}$', r'$\xi^{d}_{y}$', r'$\xi^{d}_{z}$']

        for ax, pos in zip(self.axs[0], self.positions):
            ax.set_title(f'Position[m] - {pos}')
            ax.set_xlabel('Step')
            ax.set_ylabel(pos)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        for ax, force in zip(self.axs[1], self.forces):
            ax.set_title(f'Force[N] - {force}')
            ax.set_xlabel('Step')
            ax.set_ylabel(force)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        # for ax, stiff in zip(self.axs1[2], self.stiffness):
        #     ax.set_title(f'Optimal Stiffness[N/m] - {stiff}')
        #     ax.set_xlabel('Step')
        #     ax.set_ylabel(stiff)
        #     ax.grid(True)
        #     ax.set_aspect(aspect='auto')

        # for ax, damp in zip(self.axs1[3], self.damping):
        #     ax.set_title(f'Optimal Damping[N/m] - {damp}')
        #     ax.set_xlabel('Step')
        #     ax.set_ylabel(damp)
        #     ax.grid(True)
        #     ax.set_aspect(aspect='auto')
        for ax, stiff in zip(self.axs2[0], self.stiffness):
            ax.set_title(f'Optimal Stiffness[N/m] - {stiff}')
            ax.set_xlabel('Step')
            ax.set_ylabel(stiff)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        for ax, damp in zip(self.axs2[1], self.damping):
            ax.set_title(f'Optimal Damping Ratio - {damp}')
            ax.set_xlabel('Step')
            ax.set_ylabel(damp)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.1, top=0.93, left=0.05, right=0.95, hspace=0.52, wspace=0.223)
        # plt.tight_layout()
        self.pos_list = []
        self.fd_list = []
        self.fext_list = []
        self.k_list = []
        self.d_list = []

    def update_plot(self, x, Fd, Fext, k, d, time=0.01):
        self.pos_list.append(x)
        self.fd_list.append(Fd)
        self.fext_list.append(Fext)
        self.k_list.append(k)
        self.d_list.append(d)

        for ax, data, label, color in zip(self.axs.flatten()[:3], np.array(self.pos_list).T, self.positions, ['r', 'g', 'b']):
            ax.plot(data, label=f'Position - {label}', color=color)
            if len(self.pos_list) == 1:
                ax.legend()

        # Update force plots with fill between and statistics
        for i, force in enumerate(self.forces):
            ax = self.axs[1][i]
            fd_series = np.array(self.fd_list)[:, i]
            fext_series = np.array(self.fext_list)[:, i]
            errors = fd_series - fext_series
            mean_error = np.mean(errors)
            var_error = np.var(errors)

            ax.plot(fd_series, label='Desired Force [N]', color='b')
            ax.plot(fext_series, label='Actual Force [N]', color='r',linestyle=':',linewidth=3)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series >= fext_series), color='lightgray', alpha=0.5)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series < fext_series), color='red', alpha=0.5)

            ax.set_title(f'Force - {force} [N] ($\\epsilon_{{\mu}}$: {mean_error:.2f}, $\\epsilon_{{\sigma}}^2$: {var_error:.2f})')

            if len(self.fd_list) == 1:
                ax.legend()

        # Update stiffness plots
        for ax, data, label in zip(self.axs.flatten()[6:9], np.array(self.k_list).T, self.stiffness):
            ax.plot(data, label=f'Gain - {label}', color='k')
            if len(self.k_list) == 1:
                ax.legend()

        # Update damping plots
        for ax, data, label in zip(self.axs.flatten()[9:], np.array(self.d_list).T, self.damping):
            ax.plot(data, label=f'Gain - {label}', color='k')
            if len(self.k_list) == 1:
                ax.legend()

        plt.pause(time)  # Adjust this for smoother or faster updates

    def show(self):
        plt.show()
    def save(self, title):
        # Save the plot as a file
        plt.savefig(f'{title}.png', format='png', dpi=300, bbox_inches='tight')

class VisualizerOffline:
    def __init__(self, title: str) -> None:
        # Creating subplots
        self.fig, self.axs = plt.subplots(4, 3, figsize=(30, 20))

        # Set a general title for the figure
        self.fig.suptitle(title, fontsize=14)
        self.positions = [r'$\tilde{X}$', r'$\tilde{Y}$', r'$\tilde{Z}$']
        self.forces = [r'$F_x$', r'$F_y$', r'$F_z$']
        self.stiffness = [r'$k^d_x$', r'$k^d_y$', r'$k^d_z$']
        self.damping = [r'$\xi^{d}_{x}$', r'$\xi^{d}_{y}$', r'$\xi^{d}_{z}$']

        for ax, pos in zip(self.axs[0], self.positions):
            ax.set_title(f'Position[m] - {pos}')
            ax.set_xlabel('Step')
            ax.set_ylabel(pos)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        for ax, force in zip(self.axs[1], self.forces):
            ax.set_title(f'Force[N] - {force}')
            ax.set_xlabel('Step')
            ax.set_ylabel(force)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        for ax, stiff in zip(self.axs[2], self.stiffness):
            ax.set_title(f'Optimal Stiffness[N/m] - {stiff}')
            ax.set_xlabel('Step')
            ax.set_ylabel(stiff)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        for ax, damp in zip(self.axs[3], self.damping):
            ax.set_title(f'Optimal Damping Ratio - {damp}')
            ax.set_xlabel('Step')
            ax.set_ylabel(damp)
            ax.grid(True)
            ax.set_aspect(aspect='auto')

        # self.fig1.tight_layout(rect=(0, 0.03, 1, 0.95))
        self.fig.subplots_adjust(bottom=0.1, top=0.93, left=0.05, right=0.95, hspace=0.52, wspace=0.3)




        self.pos_list = []
        self.fd_list = []
        self.fext_list = []
        self.k_list = []
        self.d_list = []

    def collect_data(self, x, Fd, Fext, k, d):
        self.pos_list.append(x)
        self.fd_list.append(Fd)
        self.fext_list.append(Fext)
        self.k_list.append(k)
        self.d_list.append(d)

    def plot_data(self):
        # Plot positions
        for ax, data, label, color in zip(self.axs.flatten()[:3], np.array(self.pos_list).T, self.positions, ['r', 'g', 'b']):
            ax.plot(data, label=f'Position - {label}', color=color)
            ax.legend()

        # Plot forces
        for i, force in enumerate(self.forces):
            ax = self.axs[1][i]
            fd_series = np.array(self.fd_list)[:, i]
            fext_series = np.array(self.fext_list)[:, i]
            errors = fd_series - fext_series
            mean_error = np.mean(errors)
            var_error = np.var(errors)

            ax.plot(fd_series, label='Desired Force [N]', color='b')
            ax.plot(fext_series, label='Actual Force [N]', color='r', linestyle=':', linewidth=3)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series >= fext_series), color='lightgray', alpha=0.5)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series < fext_series), color='red', alpha=0.5)

            ax.set_title(f'Force - {force} [N] ($\\epsilon_{{\mu}}$: {mean_error:.2f}, $\\epsilon_{{\sigma}}^2$: {var_error:.2f})')
            ax.legend()

        # Plot stiffness
        for ax, data, label in zip(self.axs.flatten()[6:9], np.array(self.k_list).T, self.stiffness):
            ax.plot(data, label=f'Gain - {label}', color='k')
            ax.legend()

        # Plot damping
        for ax, data, label in zip(self.axs.flatten()[9:], np.array(self.d_list).T, self.damping):
            ax.plot(data, label=f'Gain - {label}', color='k')
            ax.legend()

    def show(self):
        self.plot_data()
        plt.show()

    def save(self, title):
        pass
        # self.plot_data()
        # self.fig1.savefig(f'{title}_fig1.png', format='png', dpi=300, bbox_inches='tight')

# Example usage:
# visualizer = Visualizer("Optimization Results")
# During the optimization process, call visualizer.collect_data(...) to collect data
# After optimization, call visualizer.show() to display the plots
# Optionally, call visualizer.save("results") to save the plots as images


class VisualizerOfflineSameRanges:
    def __init__(self, title: str) -> None:
        # Creating subplots
        self.fig, self.axs = plt.subplots(4, 3, figsize=(30, 20))

        # Set a general title for the figure
        self.fig.suptitle(title, fontsize=14)
        self.positions = [r'$\tilde{X}$', r'$\tilde{Y}$', r'$\tilde{Z}$']
        self.forces = [r'$F_x$', r'$F_y$', r'$F_z$']
        self.stiffness = [r'$k^d_x$', r'$k^d_y$', r'$k^d_z$']
        self.damping = [r'$\xi^{d}_{x}$', r'$\xi^{d}_{y}$', r'$\xi^{d}_{z}$']

        for ax, pos in zip(self.axs[0], self.positions):
            ax.set_title(f'Position[m] - {pos}')
            ax.set_xlabel('Step')
            ax.set_ylabel(pos)
            ax.grid(True)
            # ax.set_xlim(0, 100)  # Example: Set consistent xlim for 'Step'
            ax.set_ylim(-1, 1)   # Example: Set consistent ylim for positions
            # ax.set_aspect(aspect='auto')  # Aspect ratio adjustment, if needed

        for ax, force in zip(self.axs[1], self.forces):
            ax.set_title(f'Force[N] - {force}')
            ax.set_xlabel('Step')
            ax.set_ylabel(force)
            ax.grid(True)
            # ax.set_xlim(0, 100)  # Example: Set consistent xlim for 'Step'
            ax.set_ylim(0, 100)   # Example: Set consistent ylim for forces
            # ax.set_aspect(aspect='auto')  # Aspect ratio adjustment, if needed

        for ax, stiff in zip(self.axs[2], self.stiffness):
            ax.set_title(f'Optimal Stiffness[N/m] - {stiff}')
            ax.set_xlabel('Step')
            ax.set_ylabel(stiff)
            ax.grid(True)
            # ax.set_xlim(0, 100)  # Example: Set consistent xlim for 'Step'
            ax.set_ylim(0, 5500)  # Example: Set consistent ylim for stiffness
            # ax.set_aspect(aspect='auto')  # Aspect ratio adjustment, if needed

        for ax, damp in zip(self.axs[3], self.damping):
            ax.set_title(f'Optimal Damping Ratio - {damp}')
            ax.set_xlabel('Step')
            ax.set_ylabel(damp)
            ax.grid(True)
            # ax.set_xlim(0, 100)  # Example: Set consistent xlim for 'Step'
            ax.set_ylim(0, 1.1)    # Example: Set consistent ylim for damping
            # ax.set_aspect(aspect='auto')  # Aspect ratio adjustment, if needed

        self.fig.tight_layout()
        self.fig.subplots_adjust(bottom=0.1, top=0.93, left=0.05, right=0.95, hspace=0.635, wspace=0.223)

        self.pos_list = []
        self.fd_list = []
        self.fext_list = []
        self.k_list = []
        self.d_list = []

    def collect_data(self, x, Fd, Fext, k, d):
        self.pos_list.append(x)
        self.fd_list.append(Fd)
        self.fext_list.append(Fext)
        self.k_list.append(k)
        self.d_list.append(d)

    def plot_data(self):
        # Plot positions
        for ax, data, label, color in zip(self.axs.flatten()[:3], np.array(self.pos_list).T, self.positions, ['r', 'g', 'b']):
            ax.plot(data, label=f'Position - {label}', color=color)
            ax.legend()

        # Plot forces
        for i, force in enumerate(self.forces):
            ax = self.axs[1][i]
            fd_series = np.array(self.fd_list)[:, i]
            fext_series = np.array(self.fext_list)[:, i]
            errors = fd_series - fext_series
            mean_error = np.mean(errors)
            var_error = np.var(errors)

            ax.plot(fd_series, label='Desired Force [N]', color='b')
            ax.plot(fext_series, label='Actual Force [N]', color='r', linestyle=':', linewidth=3)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series >= fext_series), color='lightgray', alpha=0.5)
            ax.fill_between(range(len(fd_series)), fd_series, fext_series, where=(fd_series < fext_series), color='red', alpha=0.5)

            ax.set_title(f'Force - {force} [N] ($\\epsilon_{{\mu}}$: {mean_error:.2f}, $\\epsilon_{{\sigma}}^2$: {var_error:.2f})')
            ax.legend()

        # Plot stiffness
        for ax, data, label in zip(self.axs.flatten()[6:9], np.array(self.k_list).T, self.stiffness):
            ax.plot(data, label=f'Gain - {label}', color='k')
            ax.legend()

        # Plot damping
        for ax, data, label in zip(self.axs.flatten()[9:], np.array(self.d_list).T, self.damping):
            ax.plot(data, label=f'Gain - {label}', color='k')
            ax.legend()

    def show(self):
        self.plot_data()
        plt.show()

    def save(self, title):
        pass
        # self.plot_data()
        # self.fig1.savefig(f'{title}_fig1.png', format='png', dpi=300, bbox_inches='tight')
