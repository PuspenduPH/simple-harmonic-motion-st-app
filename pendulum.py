import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
from pathlib import Path

from utils import (
    calculate_nonlinear_period,
    rk4_damped_pendulum,
    PALETTE,
    style_dark_subplot,
    SAVE_DIR,
)


def animate_damped_vs_undamped_pendulum_with_plots(
    L=1.0,
    G=9.81,
    theta_0_deg=20.0,
    gamma_list=[0.0, 0.25, 0.5],
    num_periods=5,
    fps=30,
    save_animation=False,
    filename=None,
    save_format="gif",
):
    """
    Animate multiple pendulums with different damping coefficients.
    All pendulums start from the same initial position.

    Parameters:
    -----------
    L : float, default=1.0
        Length of pendulum in meters
    G : float, default=9.81
        Acceleration due to gravity in m/s^2
    theta_0_deg : float, default=20.0
        Initial angle in degrees
    gamma_list : list of float, default=[0.0, 0.25, 0.5]
        List of damping coefficients for pendulums (gamma = c/m)
        First value should be 0.0 for undamped pendulum
    num_periods : int, default=5
        Number of complete periods to simulate
    fps : int, default=30
        Frames per second for animation
    save_animation : bool, default=False
        Whether to save animation as GIF
    filename : str or None, default=None
        Filename for saved GIF
    """

    theta_0 = np.radians(theta_0_deg)

    omega_0 = np.sqrt(G / L)
    T_linear = 2 * np.pi / omega_0

    T = calculate_nonlinear_period(theta_0, L, G)
    total_time = num_periods * T

    period_ratio = T / T_linear

    num_frames = int(total_time * fps) + 1
    t_array = np.linspace(0, total_time, num_frames)
    dt = t_array[1] - t_array[0]

    num_pendulums = len(gamma_list)
    theta_all = np.zeros((num_pendulums, num_frames))
    omega_all = np.zeros((num_pendulums, num_frames))

    for j in range(num_pendulums):
        theta_all[j, 0] = theta_0
        omega_all[j, 0] = 0.0

    for j in range(num_pendulums):
        for i in range(num_frames - 1):
            theta_all[j, i + 1], omega_all[j, i + 1] = rk4_damped_pendulum(
                theta_all[j, i], omega_all[j, i], dt, L, G, gamma_list[j]
            )

    x_all = L * np.sin(theta_all)
    y_all = -L * np.cos(theta_all)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(8, 7))
    fig.set_facecolor("black")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.44,
        wspace=0.28,
        left=0.1,
        right=0.95,
        top=0.86,
        bottom=0.06,
    )

    ax_pendulum = fig.add_subplot(gs[:, 0])
    ax_pendulum.set_facecolor("#091217")

    ax_position = fig.add_subplot(gs[0, 1])
    ax_velocity = fig.add_subplot(gs[1, 1])
    ax_phase = fig.add_subplot(gs[2, 1])

    ax_position.set_facecolor(PALETTE["BG_POS"])
    ax_velocity.set_facecolor(PALETTE["BG_VEL"])
    ax_phase.set_facecolor(PALETTE["BG_PHASE"])

    max_x = L * np.sin(theta_0) * 1.2

    ax_pendulum.set_xlim(-max_x, max_x)
    ax_pendulum.set_ylim(-L * 1.2, 0.3)
    style_dark_subplot(
        ax_pendulum, "Horizontal Position (m)", "Vertical Position (m)", None
    )
    ax_pendulum.axhline(0, color="#888888", linestyle="--", alpha=0.5)
    ax_pendulum.axvline(0, color="#888888", linestyle="--", alpha=0.5)

    ax_position.set_xlim(0, 1.1 * total_time)
    ax_position.set_ylim(-theta_0_deg * 1.15, theta_0_deg * 1.2)
    style_dark_subplot(
        ax_position, "Time (s)", "Angle $\\theta$ (°)", "Angular Position vs Time"
    )

    max_omega = np.max(np.abs(omega_all))
    ax_velocity.set_xlim(0, 1.1 * total_time)
    ax_velocity.set_ylim(-max_omega * 1.15, max_omega * 1.2)
    style_dark_subplot(
        ax_velocity,
        "Time (s)",
        "Angular Velocity $\\omega$ (rad/s)",
        "Angular Velocity vs Time",
    )

    ax_phase.set_xlim(-theta_0_deg * 1.15, theta_0_deg * 1.15)
    ax_phase.set_ylim(-max_omega * 1.15, max_omega * 1.2)
    style_dark_subplot(
        ax_phase,
        "Angle $\\theta$ (°)",
        "Angular Velocity $\\omega$ (rad/s)",
        "Phase Space ($\\omega$ vs $\\theta$)",
    )

    gamma_str = ", ".join([f"{g:.2f}" for g in gamma_list])
    fig.suptitle(
        f"Analysis of Undamped and Damped motion of Pendulums\n"
        f"$L$={L}m, $\\theta_0$={theta_0_deg}°, $g$={G}m/s², $\\gamma$=[{gamma_str}]",
        fontsize=13,
        fontweight="bold",
        color="white",
    )

    pivot_width = max_x / 4
    pivot_height = max_x / 16

    pivot_rect = Rectangle(
        (-pivot_width, -pivot_height),
        2 * pivot_width,
        2 * pivot_height,
        color="#696969",
        zorder=10,
        linewidth=1.5,
        edgecolor="#aaaaaa",
    )
    ax_pendulum.add_patch(pivot_rect)

    colors = PALETTE["COLORS"]
    edge_colors = PALETTE["EDGE_COLORS"]

    lines = []
    bobs = []
    trails_x = []
    trails_y = []
    trail_lines = []

    lines_pos = []
    lines_vel = []
    lines_phase = []
    points_pos = []
    points_vel = []
    points_phase = []

    for j in range(num_pendulums):
        color = colors[j % len(colors)]
        gamma_val = gamma_list[j]
        edgecolor = edge_colors[j % len(edge_colors)]

        if gamma_val == 0.0:
            label = "Undamped ($\\gamma=0$)"
        else:
            label = f"Damped ($\\gamma={gamma_val:.2f}$)"

        (line,) = ax_pendulum.plot(
            [], [], color=color, linewidth=2, alpha=0.9, label=label
        )
        bob = ax_pendulum.scatter(
            [],
            [],
            s=360,
            c=color,
            edgecolor=edgecolor,
            linewidth=1.6,
            zorder=5,
        )
        trail_x, trail_y = [], []
        (trail_line,) = ax_pendulum.plot([], [], color=color, alpha=0.45, linewidth=1.0)

        lines.append(line)
        bobs.append(bob)
        trails_x.append(trail_x)
        trails_y.append(trail_y)
        trail_lines.append(trail_line)

        (line_pos,) = ax_position.plot(
            [], [], color=color, linewidth=1, alpha=0.9, label=label
        )
        point_pos = ax_position.scatter(
            [],
            [],
            s=50,
            c=color,
            zorder=5,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        lines_pos.append(line_pos)
        points_pos.append(point_pos)

        (line_vel,) = ax_velocity.plot(
            [], [], color=color, linewidth=1, alpha=0.9, label=label
        )
        point_vel = ax_velocity.scatter(
            [],
            [],
            s=50,
            c=color,
            zorder=5,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        lines_vel.append(line_vel)
        points_vel.append(point_vel)

        (line_phase,) = ax_phase.plot(
            [], [], color=color, linewidth=1.0, alpha=0.85, label=label
        )
        point_phase = ax_phase.scatter(
            [],
            [],
            s=50,
            c=color,
            zorder=5,
            edgecolor=edgecolor,
            linewidth=1.5,
        )
        lines_phase.append(line_phase)
        points_phase.append(point_phase)

    ax_pendulum.legend(
        loc="upper left",
        fontsize=6.5,
        framealpha=0.75,
        facecolor="#1a1a2e",
        edgecolor="#555555",
        labelcolor="white",
        ncol=1 if num_pendulums <= 4 else 2,
    )

    physics_text = ax_pendulum.text(
        0.96,
        0.98,
        "",
        transform=ax_pendulum.transAxes,
        fontsize=6.5,
        va="top",
        ha="right",
        color="#FFE066",
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#1a1500",
            edgecolor="#FFCC00",
            linewidth=1.5,
            alpha=0.90,
        ),
    )

    physics_text.set_text(
        f"Pendulum Properties:\n"
        f"Length: {L:.2f} m\n"
        f"Initial angle: {theta_0_deg:.1f}°\n"
        f"Period (linear): {T_linear:.3f} s\n"
        f"Period (actual): {T:.3f} s\n"
        f"Period ratio: {period_ratio:.4f}\n"
        f"ω₀: {omega_0:.3f} rad/s"
    )

    def calculate_energy(theta, omega, m=1.0):
        """Calculate total mechanical energy"""
        kinetic = 0.5 * m * (L * omega) ** 2
        potential = m * G * L * (1 - np.cos(theta))
        return kinetic + potential

    initial_energy = calculate_energy(theta_0, 0.0)

    def animate_frame(frame):
        """Animation function called for each frame."""
        t_current = t_array[frame]
        completed_periods = t_current / T

        artists = []

        for j in range(num_pendulums):
            x = x_all[j, frame]
            y = y_all[j, frame]
            theta = theta_all[j, frame]
            omega = omega_all[j, frame]

            lines[j].set_data(np.array([0, x]), np.array([0, y]))
            bobs[j].set_offsets([[x, y]])

            trails_x[j].append(x)
            trails_y[j].append(y)
            if len(trails_x[j]) > 25:
                trails_x[j].pop(0)
                trails_y[j].pop(0)
            trail_lines[j].set_data(np.array(trails_x[j]), np.array(trails_y[j]))

            artists.extend([lines[j], bobs[j], trail_lines[j]])

            t_data = t_array[: frame + 1]
            theta_data = np.degrees(theta_all[j, : frame + 1])
            lines_pos[j].set_data(t_data, theta_data)
            points_pos[j].set_offsets([[t_current, np.degrees(theta)]])
            artists.extend([lines_pos[j], points_pos[j]])

            omega_data = omega_all[j, : frame + 1]
            lines_vel[j].set_data(t_data, omega_data)
            points_vel[j].set_offsets([[t_current, omega]])
            artists.extend([lines_vel[j], points_vel[j]])

            lines_phase[j].set_data(theta_data, omega_data)
            points_phase[j].set_offsets([[np.degrees(theta), omega]])
            artists.extend([lines_phase[j], points_phase[j]])

            # Info display omitted

        ax_pendulum.set_title(
            f"Time: {t_current:.2f} s  |  Period: {completed_periods:.2f}/{num_periods:.2f}",
            fontsize=10,
            pad=10,
            fontweight="bold",
            color="white",
        )

        return tuple(artists)

    anim = FuncAnimation(
        fig,
        animate_frame,
        frames=num_frames,
        interval=int(1000 / fps),
        blit=False,
        repeat=False,
    )

    if save_animation:
        save_format = save_format.lower().strip().lstrip(".")
        if save_format not in ("gif", "mp4"):
            print(f"Unknown format '{save_format}', defaulting to 'gif'.")
            save_format = "gif"

        gamma_str_file = "_".join([f"{g:.2f}" for g in gamma_list])
        if filename is None:
            base = (
                f"multi_pendulum_analysis_L{L}_theta{theta_0_deg:.0f}"
                f"deg_gamma{gamma_str_file}_{num_periods}periods"
            )
        else:
            base = Path(filename).stem

        full_filename = f"{base}.{save_format}"
        filepath = SAVE_DIR / full_filename

        try:
            print(
                f"Saving animation as {save_format.upper()}... This may take a moment."
            )
            if save_format == "mp4":
                writer = FFMpegWriter(
                    fps=fps,
                    codec="libx264",
                    bitrate=8000,
                    extra_args=[
                        "-pix_fmt",
                        "yuv420p",
                        "-profile:v",
                        "high",
                        "-level",
                        "4.1",
                        "-movflags",
                        "+faststart",
                    ],
                )
                anim.save(filepath, writer=writer)
            else:
                anim.save(filepath, writer="ffmpeg", fps=fps, dpi=80)
            print(f"Animation saved as: {filepath}")
        except Exception as e:
            print(f"Error saving animation: {e}")

        plt.close(fig)
    else:
        plt.show()

    print("\nMulti-Pendulum Comparison Summary:")
    print("=" * 60)
    print(f"Number of pendulums: {num_pendulums}")
    print(f"Length: {L} m")
    print(f"Initial angle: {theta_0_deg:.1f}°")
    print(f"Period (small-angle approx): {T_linear:.3f} s")
    print(f"Period (nonlinear, actual): {T:.3f} s")
    print(f"Period increase due to nonlinearity: {100 * (period_ratio - 1):.2f}%")
    print(f"Total simulation time: {total_time:.2f} s")
    print(f"Number of periods: {num_periods}")
    print(f"\nDamping coefficients: γ = {gamma_list}")
    print("\nFinal Results for Each Pendulum:")
    print("-" * 60)
    for j in range(num_pendulums):
        gamma_val = gamma_list[j]
        final_angle = np.degrees(theta_all[j, -1])
        energy_final = calculate_energy(theta_all[j, -1], omega_all[j, -1])
        energy_retention = 100 * energy_final / initial_energy

        print(f"Pendulum {j + 1} (γ={gamma_val:.3f}):")
        print(f"  Final angle: {final_angle:+.2f}° (initial: {theta_0_deg:.1f}°)")
        print(f"  Energy retention: {energy_retention:.1f}%")

    return anim


def main():
    """Main function to run the multi-pendulum animation with user inputs."""

    print("Multi-Pendulum Damping Analysis Animation")
    print("=" * 50)

    defaults = {
        "L": 1.0,
        "G": 9.81,
        "theta_0_deg": 20.0,
        "gamma_list": [0.0, 0.25, 0.5],
        "num_periods": 5,
        "fps": 30,
    }

    use_defaults = input("Use default parameters? (y/n): ").strip().lower()

    if use_defaults == "y":
        params = defaults.copy()
    else:
        params = {}
        try:
            params["L"] = float(
                input(f"Enter pendulum length in meters [{defaults['L']}]: ")
                or defaults["L"]
            )
            params["G"] = float(
                input(f"Enter gravitational acceleration in m/s² [{defaults['G']}]: ")
                or defaults["G"]
            )
            params["theta_0_deg"] = float(
                input(f"Enter initial angle in degrees [{defaults['theta_0_deg']}]: ")
                or defaults["theta_0_deg"]
            )

            gamma_input = input(
                (
                    "Enter damping coefficients as comma-separated values (e.g., 0.0,0.25,0.5) "
                    f"[{','.join(map(str, defaults['gamma_list']))}]: "
                )
            ).strip()
            if gamma_input:
                params["gamma_list"] = [
                    float(g.strip()) for g in gamma_input.split(",")
                ]
            else:
                params["gamma_list"] = defaults["gamma_list"]

            params["num_periods"] = int(
                input(
                    f"Enter number of periods to simulate [{defaults['num_periods']}]: "
                )
                or defaults["num_periods"]
            )
            params["fps"] = int(
                input(f"Enter frames per second [{defaults['fps']}]: ")
                or defaults["fps"]
            )
        except ValueError:
            print("Invalid input, using default parameters.")
            params = defaults.copy()

    save_anim = input("Save animation? (y/n): ").strip().lower() == "y"
    filename = None
    save_format = "gif"
    if save_anim:
        print("Choose save format:")
        print("  [1] GIF")
        print("  [2] MP4")
        fmt_choice = input("Enter choice (1/2) [1]: ").strip()
        if fmt_choice == "2":
            save_format = "mp4"
        else:
            save_format = "gif"

        raw_name = input(
            "Enter filename WITHOUT extension (or press Enter for auto-generated): "
        ).strip()
        if raw_name:
            filename = Path(raw_name).stem
        else:
            filename = None

    print("\nRunning simulation with parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    print(f"  save_animation: {save_anim}")
    if save_anim:
        print(f"  save_format:    {save_format}")
        print(f"  filename base:  {filename if filename else '(auto-generated)'}")

    animation = animate_damped_vs_undamped_pendulum_with_plots(
        L=params["L"],
        G=params["G"],
        theta_0_deg=params["theta_0_deg"],
        gamma_list=params["gamma_list"],
        num_periods=params["num_periods"],
        fps=params["fps"],
        save_animation=save_anim,
        filename=filename,
        save_format=save_format,
    )

    return animation


if __name__ == "__main__":
    anim = main()
