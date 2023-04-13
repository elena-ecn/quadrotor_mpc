import numpy as np
import cvxpy as cp

import utils
import config


def convex_mpc_quadrotor(A, B, Q, R, X_ref, U_ref, X_lin, U_lin, x0, u_min, u_max, N_mpc, dt):
    """Solves OCP as convex optimization problem for a time-horizon N_mpc.

    Inputs:
      - A(np.ndarray):      The discrete-time state matrix for each timestep [nxnxN_mpc]
      - B(np.ndarray):      The discrete-time input matrix for each timestep [nxmxN_mpc]
      - X_ref(np.ndarray):  The reference state trajectory to follow [nxN]
      - U_ref(np.ndarray):  The reference input trajectory to follow [mxN]
      - X_lin(np.ndarray):  The state trajectory to linearize dynamics about [nxN]
      - U_lin(np.ndarray):  The input trajectory to linearize dynamics about [mxN]
      - x0(np.ndarray):     The initial state (n,)
      - u_min(np.ndarray):  The min input bounds (m,)
      - u_max(np.ndarray):  The max input bounds (m,)
      - N_mpc(int):         The MPC time horizon
      - dt(float):          The discretization step
    Returns:
      - (np.ndarray): The first control input to be applied to the robot (m,)
    """

    n, m = B.shape[0], B.shape[1]  # State and input size

    # Decision variables
    X = cp.Variable((n, N_mpc))
    U = cp.Variable((m, N_mpc-1))

    # Objective function (quadratic)
    objective = 0
    for i in range(N_mpc-1):
        objective += 0.5*cp.quad_form(X[:, i]-X_ref[:, i], Q) + 0.5*cp.quad_form(U[:, i]-U_ref[:, i], R)

    objective += 0.5*cp.quad_form(X[:, N_mpc-1]-X_ref[:, N_mpc-1], Q)

    # Constraints
    constraints = [X[:, 0] == x0]                            # Initial condition
    for i in range(N_mpc-1):
        constraints += [u_min <= U[:, i]]                    # Control bounds
        constraints += [U[:, i] <= u_max]                    # Control bounds
        constraints += [X[:, i+1] == utils.quadrotor_rk4(X_lin[:, i], U_lin[:, i], dt) +
                        A[:, :, i]@(X[:, i]-X_lin[:, i]) + B[:, :, i]@(U[:, i]-U_lin[:, i])]  # Dynamics

    prob = cp.Problem(cp.Minimize(objective), constraints)
    prob.solve(verbose=True)
    return U.value[:, 0]


def simulation_MPC(A, B, X_ref, U_ref, X_lin_ref, U_lin_ref):
    """Simulation with MPC controller."""

    # Get data
    x0 = config.x0
    Q = config.Q
    R = config.R
    N_mpc = config.N_mpc
    N_sim = config.N_sim
    u_min = config.u_min
    u_max = config.u_max
    dt = config.dt
    n = config.Nx
    m = config.Nu

    # Simulation
    X_sim = np.zeros((n, N_sim))
    X_sim[:, 0] = x0
    U_sim = np.zeros((m, N_sim-1))

    for i in range(N_sim-1):

        # Reference trajectory for current window of N_mpc timesteps
        X_ref_tilde = X_ref[:, i:(i+N_mpc)]
        U_ref_tilde = U_ref[:, i:(i+N_mpc-1)]
        X_lin = X_lin_ref[:, i:(i+N_mpc)]
        U_lin = U_lin_ref[:, i:(i+N_mpc-1)]

        # State space matrices for the N_mpc horizon
        Ad = A[:, :, i:(i+N_mpc)]
        Bd = B[:, :, i:(i+N_mpc)]

        # Compute optimal input
        U_sim[:, i] = convex_mpc_quadrotor(Ad, Bd, Q, R, X_ref_tilde, U_ref_tilde, X_lin, U_lin, X_sim[:, i], u_min, u_max, N_mpc, dt)

        # Simulate one step
        X_sim[:, i+1] = utils.quadrotor_rk4(X_sim[:, i], U_sim[:, i], dt)

    return X_sim, U_sim
