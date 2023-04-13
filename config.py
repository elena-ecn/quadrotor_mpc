from jax.config import config
config.update("jax_enable_x64", True)   # enable float64 types for accuracy

import jax.numpy as jnp


Nx = 12             # Number of states
Nu = 4              # Number of controls
N = 250             # Timesteps of reference trajectory
N_mpc = 40          # MPC prediction horizon
dt = 0.1            # Discretization step
N_sim = 100         # Simulation timesteps
# N_sim = N + N_mpc   # Simulation timesteps

x0 = jnp.array([0.0, 0.0, 1.2, 0, 0, 0, 0, 0, 0,  0, 0, 0])  # Initial state

# Cost weights
Q = 10*jnp.eye(Nx)
R = 0.1*jnp.eye(Nu)

# Input bounds
u_min = jnp.zeros(Nu)
u_max = 10*jnp.ones(Nu)
