from jax.config import config
config.update("jax_enable_x64", True)   # enable float64 types for accuracy

import numpy as np
import jax.numpy as jnp
from jax import jit, jacfwd
import pandas as pd

import config


@jit
def skew(v):
    """Computes the skew-symmetric matrix that converts a vector cross product to matrix multiplication.

    axb = skew(a)*b
    Inputs:
      - v(np.ndarray): A 3D vector [3x1]
    Returns:
      - (jnp.ndarray): The skew-symmetric matrix [3x3]
    """
    v = v.reshape(3,)  # To ensure both [3,1] and (3,) vectors work
    return jnp.array([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])


@jit
def dcm_from_mrp(p):
    """Converts Modified Rodrigues Parameters (MRP) to Direction Cosine Matrix (DCM).

    Inputs:
      - p(np.ndarray):  The Modified Rodrigues Parameters (MRP) [3x1]
    Returns:
      - R(jnp.ndarray): The Direction Cosine Matrix (DCM) [3x3]
    """
    p1 = p[0][0]
    p2 = p[1][0]
    p3 = p[2][0]

    den = (p1**2 + p2**2 + p3**2 + 1)**2
    a = (4*p1**2 + 4*p2**2 + 4*p3**2 - 4)
    R = jnp.array([[-((8*p2**2+8*p3**2)/den-1)*den, 8*p1*p2 + p3*a, 8*p1*p3 - p2*a],
                   [8*p1*p2 - p3*a, -((8*p1**2 + 8*p3**2)/den - 1)*den, 8*p2*p3 + p1*a],
                   [8*p1*p3 + p2*a, 8*p2*p3 - p1*a, -((8*p1**2 + 8*p2**2)/den - 1)*den]])/den
    return R


def quadrotor_dynamics(x, u):
    """Computes the continuous-time dynamics for a quadrotor ẋ=f(x,u).

    State is x = [r, v, p, omega], where:
    - r ∈R^3 is the position in world frame N
    - v ∈R^3 is the linear velocity in world frame N
    - p ∈R^3 is the attitude from B->N (MRP)
    - omega ∈R^3 is the angular velocity in body frame B
    Inputs:
      - x(np.ndarray): The system state   [12x1]
      - u(np.ndarray): The control inputs [4x1]
    Returns:
      - x_d(np.ndarray): The time derivative of the state [12x1]
    """
    # Quadrotor parameters
    mass = 0.5
    L = 0.1750
    J = jnp.diag(jnp.array([0.0023, 0.0023, 0.004]))
    kf = 1.0
    km = 0.0245
    gravity = jnp.array([0,0,-9.81]).reshape(3, 1)

    # State variables
    r = x[0:3].reshape(3, 1)
    v = x[3:6].reshape(3, 1)
    p = x[6:9].reshape(3, 1)
    omega = x[9:12].reshape(3, 1)

    Q = dcm_from_mrp(p)

    w1 = u[0]
    w2 = u[1]
    w3 = u[2]
    w4 = u[3]

    F1 = max(0, kf*w1)
    F2 = max(0, kf*w2)
    F3 = max(0, kf*w3)
    F4 = max(0, kf*w4)
    F = jnp.array([0.0, 0.0, F1+F2+F3+F4]).reshape(3, 1)  # Total rotor force in body frame

    M1 = km*w1
    M2 = km*w2
    M3 = km*w3
    M4 = km*w4
    tau = jnp.array([L*(F2-F4), L*(F3-F1), (M1-M2+M3-M4)]).reshape(3, 1)  # Total rotor torque in body frame

    f = mass*gravity + Q@F  # Forces in world frame

    # Dynamics
    r_d = v
    v_d = f/mass
    p_d = ((1+jnp.linalg.norm(p)**2)/4)*(jnp.eye(3) + 2*(skew(p)@skew(p)+skew(p))/(1+jnp.linalg.norm(p)**2))@omega
    cross_pr = jnp.cross(omega.reshape(3,), (J@omega).reshape(3,)).reshape(3,1)
    omega_d, _, _, _ = jnp.linalg.lstsq(J, tau - cross_pr, rcond=None)

    return jnp.vstack((r_d, v_d, p_d, omega_d)).reshape(12,)  # x_dot


def quadrotor_rk4(x, u, Ts):
    """Discrete-time dynamics: Integration with RK4 method."""
    f = quadrotor_dynamics
    k1 = Ts*f(x, u)
    k2 = Ts*f(x + k1/2, u)
    k3 = Ts*f(x + k2/2, u)
    k4 = Ts*f(x + k3, u)
    x_next = x + 1/6*(k1 + 2*k2 + 2*k3 + k4)
    return x_next


def get_linearized_dynamics_matrices(X_ref, U_ref, Ts):
    """Linearizes dynamics about reference trajectory.

    Inputs:
      - X_ref(np.ndarray): The state reference trajectory    [12xN]
      - U_ref(np.ndarray): The controls reference trajectory [4xN]
    Returns:
      - Ad(np.ndarray): The linearized state matrix at each timestep     [12x12xN]
      - Bd(np.ndarray): The linearized controls matrix at each timestep  [12x4xN]
    """
    Nx = X_ref.shape[0]
    Nu = U_ref.shape[0]
    N_mpc = X_ref.shape[1]

    Ad = np.zeros((Nx, Nx, N_mpc))
    Bd = np.zeros((Nx, Nu, N_mpc))
    for i in range(N_mpc-1):
        Ad[:, :, i] = jacfwd(quadrotor_rk4, 0)(X_ref[:, i], U_ref[:, i], Ts)  # [12x12]
        Bd[:, :, i] = jacfwd(quadrotor_rk4, 1)(X_ref[:, i], U_ref[:, i], Ts)  # [12x4]
    return Ad, Bd


def get_ref_trajectory():
    """Computes the reference trajectories."""
    x0 = config.x0
    Nx = config.Nx
    Nu = config.Nu
    N = config.N
    dt = config.dt

    # Trajectory to linearize dynamics about
    X_lin_ref = jnp.tile(x0.reshape(Nx, 1), (1, N))  # nxN
    U_lin_ref = jnp.tile((9.81*0.5/4)*jnp.ones(Nu).reshape(Nu, 1), (1, N - 1))  # mx(N-1)

    # Reference trajectory to follow
    U_ref = U_lin_ref
    X_ref = np.zeros((Nx, N))
    i = 0
    for t in np.linspace(-np.pi/2, 3*np.pi/2 + 4*np.pi, N):
        X_ref[:, i] = np.hstack(
            (np.array([5*np.cos(t), 5*np.cos(t)*np.sin(t), 1.2]), np.zeros(3), 1e-9*np.ones(3), np.zeros(3)))
        i += 1
    for i in range(N - 1):
        X_ref[3:6, i] = (X_ref[0:3, i + 1] - X_ref[0:3, i])/dt

    return X_ref, U_ref, X_lin_ref, U_lin_ref


def save_to_file(x_history, X_ref):
    """Save state trajectory to csv file to visualize with Julia."""

    # Save trajectory
    filename = "X_quadrotor.csv"
    df = pd.DataFrame(x_history)   # convert array into dataframe
    df.to_csv(filename, index=False, header=False, float_format='%f')

    # Save reference trajectory
    filename = "X_ref.csv"
    df = pd.DataFrame(X_ref)  # convert array into dataframe
    df.to_csv(filename, index=False, header=False, float_format='%f')
