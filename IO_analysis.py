     # visualize time per step
        plt.plot(convergence_time_per_step)
        plt.xlabel('Step')
        plt.ylabel('Time [s]')
        plt.show()

        axs = {0: 'X', 1: 'Y', 2: 'Z'} # Axis labels
        # get F_d norm bound threshold
        F_d_bound = get_norm_bound_threshold(force)
        print(f"F_d_bound: {F_d_bound}")

        # get F_d continuity threshold (critertion value)
        F_d_continuity = get_lipschitz_criterion(force)
        print(f"F_d_continuity: {F_d_continuity}")
        # visualize F_d continuity
        F_d = np.array(force).T
        for ax, i in enumerate(range(3)):
            plt.plot(F_d[i], label=f"{axs[ax]}")
        plt.title(f"F_d: generated force on X,Y and Z axis - {MATERIAL_NAME} material")
        plt.legend()
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ MATERIAL_NAME)
        plt.show()

        # get F_d smoothness threshold: derivatives of F_d are bounded
        F_d_sm = get_smoothness_threshold(force)
        print(f"F_d_sm: {F_d_sm}")

        # get the continuity of F_d_dot
        F_d_dot = np.diff(force, axis=0)
        F_d_dot_array = np.array(F_d_dot).T
        print(f"F_d_dot: {F_d_dot}")
        for ax, i in enumerate(range(3)):
            plt.plot(F_d_dot_array[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{F_d} :1^{st} derivative generated force on X,Y and Z$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{F_d}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()

        F_d_dot_continuity = get_lipschitz_criterion(F_d_dot)
        print(f"F_d_dot_continuity: {F_d_dot_continuity}")

        # get the continuity of F_d_ddot: second derivative of F_d is continuous
        F_d_ddot = np.diff(F_d_dot, axis=0)
        F_d_ddot_array = np.array(F_d_ddot).T
        print(f"F_d_ddot: {F_d_ddot}")
        for ax, i in enumerate(range(3)):
            plt.plot(F_d_ddot_array[i], label=f"{axs[ax]}")
        plt.title(r'$\ddot{F_d}:2^{nd} derivative generated force on X,Y and Z$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$F_d$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()
        F_ddot_continuity = get_lipschitz_criterion(F_d_ddot)

        # get the continuity of X_tilde and X_tilde_dot
        X = np.array(x_tilde_list).T
        X_dot = np.array(x_tilde_dot_list).T
        x_tilde_continuity = get_lipschitz_criterion(X.T)
        print(f"x_tilde_continuity: {x_tilde_continuity}")
        x_tilde_dot_continuity = get_lipschitz_criterion(X_dot.T)
        print(f"x_tilde_dot_continuity: {x_tilde_dot_continuity}")
        x_dot_bound = get_norm_bound_threshold(X_dot.T)
        print(f"x_dot_bound: {x_dot_bound}")
        x_tilde_bound = get_norm_bound_threshold(X.T)
        print(f"x_tilde_bound: {x_tilde_bound}")
        for ax, i in enumerate(range(3)):
            plt.plot(X[i], label=f"{axs[ax]}")
        plt.title(r'$\tilde{x}: position$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\tilde{x}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()

        for ax, i in enumerate(range(3)):
            plt.plot(X_dot[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()
        # visualise dissipated energy
        plt.plot(controller.E_tot)
        plt.xlabel('Step')
        plt.ylabel(r'Tank storage $T(x_t)$: '+ MATERIAL_NAME)
        plt.show()

        # visualise velocity error ZYZ
        for ax, i in enumerate(range(3)):
            plt.plot(x_tilde_dot_list[i], label=f"{axs[ax]}")
        plt.title(r'$\dot{\tilde{x}}: velocity error$: '+ MATERIAL_NAME)
        plt.xlabel('Step')
        plt.ylabel(r'$\dot{\tilde{x}}$: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()

        norm_bounds = []
        for x in controller.Force_error:
            norm_bounds.append(x[-1])
        plt.plot(norm_bounds)
        plt.xlabel('Step')
        plt.ylabel(r'Force error norm: '+ MATERIAL_NAME)
        plt.show()

        for i in range(4):
            rand_idx = np.random.randint(0, len(controller.Force_error))
            plt.plot(controller.Force_error[rand_idx], label=f"Step {rand_idx}")
        plt.plot(controller.Force_error[-1], label=f"Step {len(controller.Force_error)}")
        plt.xlabel('optimizer iterations per step')
        plt.ylabel(r'Force error: '+ MATERIAL_NAME)
        plt.legend()
        plt.show()


        plt.plot(controller.fun_value)
        plt.xlabel('Step')
        plt.ylabel(r'optimizer fun_value: '+ MATERIAL_NAME)
        plt.show()
        convergence_delta = np.max(controller.step_Force_errors)
        print(f"convergence_delta: {convergence_delta}")