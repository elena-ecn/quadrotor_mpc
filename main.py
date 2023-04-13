"""
Quadrotor trajectory tracking control with Model Predictive Control (MPC).

Author: Elena Oikonomou
Date: Spring 2023
"""
import utils
import mpc
import plotter
import config


def main():
    # Trajectory tracking with MPC
    X_ref, U_ref, X_lin_ref, U_lin_ref = utils.get_ref_trajectory()
    A, B = utils.get_linearized_dynamics_matrices(X_ref, U_ref, config.dt)  # nxnxN, nxmxN
    x_history, u_history = mpc.simulation_MPC(A, B, X_ref, U_ref, X_lin_ref, U_lin_ref)

    # Plot trajectories (position & linear velocities)
    plotter.plot_trajectories_pos(x_history, X_ref)
    plotter.plot_trajectories_vel(x_history, X_ref)

    # Save data for visualization
    utils.save_to_file(x_history)


if __name__ == "__main__":
    main()
