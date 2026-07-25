"""
utils.py - Shared Utility and Helper Functions for Simple Harmonic Motion
=======================================================================
"""

from pathlib import Path
import numpy as np
from scipy.special import ellipk

SCRIPT_PATH = Path(__file__).resolve().parent
SAVE_DIR = SCRIPT_PATH / "outputs"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

PALETTE: dict = {
    "BG_MAIN": "#091217",
    "BG_POS": "#00001A",
    "BG_VEL": "#110000",
    "BG_PHASE": "#001100",
    "BG_ENERGY": "#110000",
    "GRID": "#444444",
    "GUIDE": "#555555",
    "COLORS": [
        "#00EAFF",
        "#AAFF00",
        "#FF4D6D",
        "#FF9F1C",
        "#BF5AF2",
        "#FF2EF7",
        "#FFD700",
    ],
    "EDGE_COLORS": [
        "#008899",
        "#668800",
        "#991133",
        "#996010",
        "#7B1FA2",
        "#991199",
        "#B8860B",
    ],
}


def style_dark_subplot(
    ax,
    xlabel,
    ylabel,
    title,
    grid_color=PALETTE["GRID"],
    grid_alpha=0.3,
    label_fontsize=8,
    title_fontsize=10,
    tick_labelsize=7,
):
    """Apply consistent dark-theme styling to a subplot."""
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=label_fontsize, color="white")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=label_fontsize, color="white")
    if title:
        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", color="white")
    ax.tick_params(colors="white", which="both", labelsize=tick_labelsize)
    for sp in ax.spines.values():
        sp.set_edgecolor(PALETTE["GUIDE"])
    ax.grid(True, alpha=grid_alpha, color=grid_color)


# ===========================================================================
# Mass-Spring Systems
# ===========================================================================


def mass_spring_damper_system(t, state, m, c, k, forcing_func):
    x, v = state
    F_t = forcing_func(t)
    dxdt = v
    dvdt = (F_t - c * v - k * x) / m
    return np.array([dxdt, dvdt])


def rk4_step(t, y, h, m, c, k, forcing_func):
    k1 = h * mass_spring_damper_system(t, y, m, c, k, forcing_func)
    k2 = h * mass_spring_damper_system(t + h / 2, y + k1 / 2, m, c, k, forcing_func)
    k3 = h * mass_spring_damper_system(t + h / 2, y + k2 / 2, m, c, k, forcing_func)
    k4 = h * mass_spring_damper_system(t + h, y + k3, m, c, k, forcing_func)
    return y + (k1 + 2 * k2 + 2 * k3 + k4) / 6


def rk4_mass_spring_damper(m, c, k, x0, v0, t0, tf, h, forcing_func):
    n_steps = int((tf - t0) / h) + 1
    t_vals = np.linspace(t0, tf, n_steps)
    x_vals = np.zeros(n_steps)
    v_vals = np.zeros(n_steps)
    x_vals[0] = x0
    v_vals[0] = v0
    state = np.array([x0, v0])
    for i in range(n_steps - 1):
        state = rk4_step(t_vals[i], state, h, m, c, k, forcing_func)
        x_vals[i + 1] = state[0]
        v_vals[i + 1] = state[1]
    return t_vals, x_vals, v_vals


def no_forcing(t):
    return 0.0


def step_forcing(t, step_time=0.0, amplitude=1.0):
    return amplitude if t >= step_time else 0.0


def sinusoidal_forcing(t, amplitude=1.0, frequency=1.0):
    return amplitude * np.sin(2 * np.pi * frequency * t)


def impulse_forcing(t, impulse_time=0.0, amplitude=10.0, duration=0.1):
    if impulse_time <= t <= impulse_time + duration:
        return amplitude
    return 0.0


def analyze_system_properties(m, c, k):
    omega_n = np.sqrt(k / m)
    zeta = c / (2 * np.sqrt(k * m))
    properties = {
        "omega_n": omega_n,
        "zeta": zeta,
        "period": 2 * np.pi / omega_n if omega_n > 0 else float("inf"),
    }
    if zeta < 1:
        properties["damping_type"] = "underdamped"
        properties["omega_d"] = omega_n * np.sqrt(1 - zeta**2)
    tol = 1e-3
    if abs(zeta - 1.0) <= tol:
        properties["damping_type"] = "critically damped"
        properties.pop("omega_d", None)
    elif zeta > 1.0:
        properties["damping_type"] = "overdamped"
    return properties


# ===========================================================================
# Pendulum Systems
# ===========================================================================


def calculate_nonlinear_period(theta_0, L, G):
    T_0 = 2 * np.pi * np.sqrt(L / G)
    k = np.sin(theta_0 / 2)
    K_k = ellipk(k**2)
    T_actual = T_0 * (2 / np.pi) * K_k
    return T_actual


def rk4_damped_pendulum(theta, omega, dt, L, G, gamma):
    def derivatives(theta, omega):
        dtheta_dt = omega
        domega_dt = -(G / L) * np.sin(theta) - gamma * omega
        return dtheta_dt, domega_dt

    k1_theta, k1_omega = derivatives(theta, omega)
    k2_theta, k2_omega = derivatives(
        theta + 0.5 * dt * k1_theta, omega + 0.5 * dt * k1_omega
    )
    k3_theta, k3_omega = derivatives(
        theta + 0.5 * dt * k2_theta, omega + 0.5 * dt * k2_omega
    )
    k4_theta, k4_omega = derivatives(theta + dt * k3_theta, omega + dt * k3_omega)
    theta_new = theta + (dt / 6.0) * (k1_theta + 2 * k2_theta + 2 * k3_theta + k4_theta)
    omega_new = omega + (dt / 6.0) * (k1_omega + 2 * k2_omega + 2 * k3_omega + k4_omega)
    return theta_new, omega_new
