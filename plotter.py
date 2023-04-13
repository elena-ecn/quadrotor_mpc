import matplotlib.pyplot as plt
import seaborn as sns


def plot_trajectories_pos(x_history, x_des):
    """Plots state trajectories. (Positions)"""

    N = x_history.shape[1]

    # Plot state trajectories
    sns.set_theme()
    plt.figure()
    plt.plot(x_history[0, :], label="x1")
    plt.plot(x_history[1, :], label="x2")
    plt.plot(x_history[2, :], label="x3")
    plt.plot(x_des[0, :N], 'b--', label="x1_des")
    plt.plot(x_des[1, :N], 'r--', label="x2_des")
    plt.plot(x_des[2, :N], 'g--', label="x3_des")
    plt.xlabel("N")
    plt.legend()
    plt.title("State trajectories - Positions")
    plt.savefig('images/state_trajectories_pos.png')
    plt.show()


def plot_trajectories_vel(x_history, x_des):
    """Plots state trajectories. (Velocities)"""

    N = x_history.shape[1]

    # Plot state trajectories
    sns.set_theme()
    plt.figure()
    plt.plot(x_history[3, :], label="vx")
    plt.plot(x_history[4, :], label="vy")
    plt.plot(x_history[5, :], label="vz")
    plt.plot(x_des[3, :N], 'b--', label="vx_des")
    plt.plot(x_des[4, :N], 'r--', label="vy_des")
    plt.plot(x_des[5, :N], 'g--', label="vz_des")
    plt.xlabel("N")
    plt.legend()
    plt.title("State trajectories - Velocities")
    plt.savefig('images/state_trajectories_vel.png')
    plt.show()
