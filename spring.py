from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.gridspec import GridSpec

from utils import (
    rk4_mass_spring_damper,
    no_forcing,
    analyze_system_properties,
    PALETTE,
    style_dark_subplot,
    SAVE_DIR,
)


def animate_multiple_damping_cases_with_plots(
    m=1.0,
    c_values=[0, 0.5],
    k=10.0,
    x0=0.1,
    v0=0.0,
    oscillations=5.0,
    fps=30,
    save_animation=False,
    filename=None,
    save_format="gif",
    num_coils=8,
    vertical_separation=0.15,
    forcing_func=None,
    forcing_name="none",
):
    """
    Animate multiple damping cases (undamped, underdamped, critically damped, overdamped) simultaneously.

    Parameters:
        m: Mass
        c_values: List of damping coefficients (0 for undamped, then underdamped, critical, overdamped)
        k: Stiffness
        x0: Initial displacement
        v0: Initial velocity
        oscillations: Number of oscillations to show
        fps: Frames per second
        save_animation: Whether to save as GIF
        filename: Output filename
        num_coils: Number of coils in spring visualization
        vertical_separation: Vertical distance between systems
        forcing_func: External forcing function
        forcing_name: Name of forcing function for display
    """
    if forcing_func is None:
        forcing_func = no_forcing

    # System properties
    omega_n = np.sqrt(k / m)
    T = 2 * np.pi / omega_n
    c_critical = 2 * np.sqrt(k * m)

    # Time parameters
    duration = oscillations * T
    num_frames = int(round(duration * fps)) + 1
    t_array = np.linspace(0.0, duration, num_frames, endpoint=True)
    h = duration / (num_frames - 1)

    num_cases = len(c_values)
    x_solutions = []
    v_solutions = []
    case_labels = []
    case_colors = PALETTE["COLORS"]
    edge_colors = PALETTE["EDGE_COLORS"]

    for i, c in enumerate(c_values):
        if c == 0:
            x_vals = x0 * np.cos(omega_n * t_array)
            v_vals = -x0 * omega_n * np.sin(omega_n * t_array)
            case_labels.append("Undamped (c=0)")
        else:
            _, x_vals, v_vals = rk4_mass_spring_damper(
                m, c, k, x0, v0, 0.0, duration, h, forcing_func
            )
            props = analyze_system_properties(m, c, k)
            case_labels.append(f"{props['damping_type'].title()} (c={c:.2f})")

        x_solutions.append(x_vals)
        v_solutions.append(v_vals)

    def calculate_energy(x, v, mass=m, stiffness=k):
        """Calculate total mechanical energy"""
        kinetic = 0.5 * mass * v**2
        potential = 0.5 * stiffness * x**2
        return kinetic + potential

    energy_solutions = []
    for x_vals, v_vals in zip(x_solutions, v_solutions):
        E_vals = calculate_energy(x_vals, v_vals)
        energy_solutions.append(E_vals)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(10, 8))
    fig.set_facecolor("black")
    gs = GridSpec(
        3,
        2,
        figure=fig,
        hspace=0.4,
        wspace=0.26,
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.1,
        width_ratios=[1.5, 1],
    )

    ax = fig.add_subplot(gs[:, 0])
    ax.set_facecolor(PALETTE["BG_MAIN"])

    ax_position = fig.add_subplot(gs[0, 1])
    ax_phase = fig.add_subplot(gs[1, 1])
    ax_energy = fig.add_subplot(gs[2, 1])

    ax_position.set_facecolor(PALETTE["BG_POS"])
    ax_phase.set_facecolor(PALETTE["BG_PHASE"])
    ax_energy.set_facecolor(PALETTE["BG_ENERGY"])

    c_str = ", ".join([str(c) for c in c_values])
    fig.suptitle(
        f"Mass-Spring-Damper System Analysis: Multiple Damping Cases\n"
        f"$m$={m} kg, $k$={k} N/m, $c_{{crit}}$={c_critical:.2f} N·s/m, "
        f"$\\omega_0$={omega_n:.2f} rad/s, Forcing: {forcing_name}",
        fontsize=13,
        fontweight="bold",
        color="white",
    )

    wall_x = -1.5 * x0
    total_height = (num_cases - 1) * vertical_separation

    ax.set_xlim(wall_x - 0.02, 1.6 * abs(x0))
    ax.set_ylim(-total_height - 0.15, 0.15)
    ax.set_xlabel("Position (m)", fontsize=10, color="white")
    ax.set_ylabel("")
    ax.set_yticks([])
    ax.tick_params(colors="white", which="both", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#555555")
    ax.grid(True, alpha=0.2, color="#444444")

    ax.axvline(wall_x, color="#555555", linewidth=15, zorder=4)

    ax.axvline(0, color="#888888", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(x0, color="#888888", linestyle="--", alpha=0.4, linewidth=1)
    ax.axvline(-x0, color="#888888", linestyle="--", alpha=0.4, linewidth=1)

    for i in range(num_cases):
        y_pos = -i * vertical_separation
        ax.axhline(y_pos, color="#555555", alpha=0.4, linewidth=0.5)

    masses = []
    springs = []
    lines_pos = []
    points_pos = []
    lines_phase = []
    points_phase = []
    lines_energy = []
    points_energy = []

    for i, (c, label) in enumerate(zip(c_values, case_labels)):
        color = case_colors[i % len(case_colors)]
        edge_color = edge_colors[i % len(edge_colors)]

        mass = ax.scatter(
            [],
            [],
            s=360,
            c=color,
            marker="s",
            edgecolor=edge_color,
            linewidth=1.5,
            label=label,
            zorder=5,
        )
        masses.append(mass)

        (spring,) = ax.plot([], [], color=color, lw=2, alpha=0.95, zorder=3)
        springs.append(spring)

        (line_pos,) = ax_position.plot(
            [], [], color=color, linewidth=1, alpha=0.9, label=label
        )
        point_pos = ax_position.scatter(
            [], [], s=50, c=color, zorder=5, edgecolor=edge_color, linewidth=1
        )
        lines_pos.append(line_pos)
        points_pos.append(point_pos)

        (line_phase,) = ax_phase.plot(
            [], [], color=color, linewidth=1, alpha=0.85, label=label
        )
        point_phase = ax_phase.scatter(
            [], [], s=50, c=color, zorder=5, edgecolor=edge_color, linewidth=1
        )
        lines_phase.append(line_phase)
        points_phase.append(point_phase)

        (line_energy,) = ax_energy.plot(
            [], [], color=color, linewidth=1, alpha=0.9, label=label
        )
        point_energy = ax_energy.scatter(
            [], [], s=50, c=color, zorder=5, edgecolor=edge_color, linewidth=1
        )
        lines_energy.append(line_energy)
        points_energy.append(point_energy)

    ax.legend(
        loc="upper right",
        fontsize=6.5,
        fancybox=True,
        facecolor="#1a1a2e",
        edgecolor="#555555",
        labelcolor="white",
        markerscale=0.4,
        labelspacing=1.1,
    )

    # Configuration for position vs time plot
    max_x = max(np.max(np.abs(x)) for x in x_solutions)
    ax_position.set_xlim(0, 1.1 * duration)
    ax_position.set_ylim(-max_x * 1.15, max_x * 1.2)
    style_dark_subplot(
        ax_position,
        "Time (s)",
        "Position $x$ (m)",
        "Position vs Time",
        label_fontsize=9,
        tick_labelsize=8,
    )

    # Configuration for phase space plot
    max_v = max(np.max(np.abs(v)) for v in v_solutions)
    ax_phase.set_xlim(-max_x * 1.15, max_x * 1.15)
    ax_phase.set_ylim(-max_v * 1.15, max_v * 1.2)
    style_dark_subplot(
        ax_phase,
        "Position $x$ (m)",
        "Velocity $v$ (m/s)",
        "Phase Space ($v$ vs $x$)",
        label_fontsize=9,
        tick_labelsize=8,
    )

    # Configuration for energy vs time plot
    max_E = max(np.max(E) for E in energy_solutions)
    ax_energy.set_xlim(0, 1.1 * duration)
    ax_energy.set_ylim(-0.05 * max_E, max_E * 1.2)
    style_dark_subplot(
        ax_energy,
        "Time (s)",
        "Energy $E$ (J)",
        "Total Energy vs Time",
        label_fontsize=9,
        tick_labelsize=8,
    )

    def draw_spring_horizontal(start_x, end_x, num_coils, radius=0.03, y_center=0.0):
        """Draw a horizontal spring from start_x to end_x"""
        if abs(end_x - start_x) < 1e-9:
            return np.array([start_x]), np.array([y_center])

        t = np.linspace(0, num_coils * 2 * np.pi, 300)
        x_spring = np.linspace(start_x, end_x, 300)
        y_spring = y_center + radius * np.sin(t)

        return x_spring, y_spring

    def init():
        """Initialize animation"""
        for i, (mass, spring) in enumerate(zip(masses, springs)):
            y_pos = -i * vertical_separation
            mass.set_offsets(np.array([[np.nan, y_pos]]))
            spring.set_data([], [])
        for line_pos, line_phase, line_energy in zip(
            lines_pos, lines_phase, lines_energy
        ):
            line_pos.set_data([], [])
            line_phase.set_data([], [])
            line_energy.set_data([], [])
        return tuple(
            masses
            + springs
            + lines_pos
            + points_pos
            + lines_phase
            + points_phase
            + lines_energy
            + points_energy
        )

    def update(frame):
        """Update animation for each frame"""
        t = t_array[frame]
        num_osc = min(t / T, oscillations)

        ax.set_title(
            f"Time: {t:.2f} s  |  Oscillation: {num_osc:.2f}/{oscillations:.2f}",
            fontsize=10,
            fontweight="bold",
            color="white",
            pad=10,
        )

        artists = []

        for i, (c, x_vals, v_vals, E_vals) in enumerate(
            zip(c_values, x_solutions, v_solutions, energy_solutions)
        ):
            y_pos = -i * vertical_separation
            x_current = x_vals[frame]
            v_current = v_vals[frame]
            E_current = E_vals[frame]

            x_spring, y_spring = draw_spring_horizontal(
                wall_x, x_current, num_coils, radius=0.03, y_center=y_pos
            )
            springs[i].set_data(x_spring, y_spring)
            masses[i].set_offsets([[x_current, y_pos]])

            t_data = t_array[: frame + 1]
            x_data = x_vals[: frame + 1]
            lines_pos[i].set_data(t_data, x_data)
            points_pos[i].set_offsets([[t, x_current]])

            v_data = v_vals[: frame + 1]
            lines_phase[i].set_data(x_data, v_data)
            points_phase[i].set_offsets([[x_current, v_current]])

            E_data = E_vals[: frame + 1]
            lines_energy[i].set_data(t_data, E_data)
            points_energy[i].set_offsets([[t, E_current]])

            artists.extend(
                [
                    masses[i],
                    springs[i],
                    lines_pos[i],
                    points_pos[i],
                    lines_phase[i],
                    points_phase[i],
                    lines_energy[i],
                    points_energy[i],
                ]
            )

        return tuple(artists)

    anim = FuncAnimation(
        fig,
        update,
        frames=num_frames,
        init_func=init,
        blit=False,
        interval=int(1000.0 / fps),
        repeat=False,
    )

    if save_animation:
        save_format = save_format.lower().strip().lstrip(".")
        if save_format not in ("gif", "mp4"):
            print(f"Unknown format '{save_format}', defaulting to 'gif'.")
            save_format = "gif"

        c_str = "_".join([f"c{c}" for c in c_values])
        if filename is None:
            base = f"multi_damping_analysis_m{m}_k{k}_{c_str}"
        else:
            base = Path(filename).stem

        full_filename = f"{base}.{save_format}"
        filepath = SAVE_DIR / full_filename

        try:
            print(
                f"Saving animation as {save_format.upper()} to {filepath.resolve()}..."
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
                anim.save(filepath, writer="ffmpeg", fps=fps, dpi=120)
            print(f"Animation saved as: {filepath}")
        except Exception as e:
            print(f"Error saving animation: {e}")
        plt.close(fig)
    else:
        plt.show()

    return anim


def main():
    """Main function to run the multi-damping animation with user inputs."""

    print("\n" + "=" * 70)
    print("  Mass-Spring-Damper Animation: Multiple Damping Cases")
    print("=" * 70)
    print("\nThis animation compares multiple damping cases:")
    print("  - Undamped (c=0)")
    print("  - Underdamped (0 < c < c_critical)")
    print("  - Critically damped (c = c_critical)")
    print("  - Overdamped (c > c_critical)\n")

    defaults = {
        "m": 1.0,
        "k": 10.0,
        "c_values": [0, 0.5, 6.32, 7],
        "x0": 0.1,
        "v0": 0.0,
        "oscillations": 5.0,
        "fps": 30,
    }

    use_defaults = input("Use default parameters? (y/n) [y]: ").strip().lower()

    if use_defaults == "n":
        try:
            print("\n--- System Parameters ---")
            m = float(input("Mass (m) [1.0 kg]: ") or 1.0)
            k = float(input("Stiffness (k) [10.0 N/m]: ") or 10.0)

            c_critical = 2 * np.sqrt(k * m)
            print(f"\nCritical damping coefficient: c_crit = {c_critical:.3f} N·s/m")

            print("\n--- Damping Coefficients ---")
            print("Enter damping coefficients (comma-separated).")
            print("Include 0 for undamped case.")
            print(
                f"Suggestions: 0 (undamped), {c_critical * 0.2:.2f} (underdamped), {c_critical:.2f} (critical), {c_critical * 1.5:.2f} (overdamped)"
            )
            c_input = input(
                f"c values [0, {c_critical * 0.2:.2f}, {c_critical:.2f}, {c_critical * 1.5:.2f}]: "
            ).strip()

            if c_input:
                c_values = [float(c.strip()) for c in c_input.split(",")]
            else:
                c_values = [0, c_critical * 0.2, c_critical, c_critical * 1.5]

            c_values = sorted(c_values)

            x0 = float(input("Initial displacement/amplitude (x0) [0.1 m]: ") or 0.1)
            v0 = float(input("Initial velocity v0 [0.0 m/s]: ") or 0.0)

            print("\n--- Animation Parameters ---")
            oscillations = float(input("Number of periods to display [5.0]: ") or 5.0)
            fps = int(input("Frames per second [30]: ") or 30)

        except ValueError as e:
            print(f"\nInvalid input: {e}")
            print("Using default parameters.")
            m = defaults["m"]
            k = defaults["k"]
            c_values = defaults["c_values"]
            x0 = defaults["x0"]
            v0 = defaults["v0"]
            oscillations = defaults["oscillations"]
            fps = defaults["fps"]
    else:
        m = defaults["m"]
        k = defaults["k"]
        c_values = defaults["c_values"]
        x0 = defaults["x0"]
        v0 = defaults["v0"]
        oscillations = defaults["oscillations"]
        fps = defaults["fps"]

    if m <= 0:
        print("Warning: Mass must be positive, setting to 1.0")
        m = 1.0
    if k <= 0:
        print("Warning: Stiffness must be positive, setting to 10.0")
        k = 10.0
    if any(c < 0 for c in c_values):
        print("Warning: Damping coefficients must be non-negative")
        c_values = [max(0, c) for c in c_values]

    forcing_func = no_forcing
    forcing_name = "None"

    save_anim = input("\nSave animation? (y/n) [n]: ").strip().lower() == "y"
    filename = None
    save_format = "gif"
    if save_anim:
        print("Choose save format:")
        print("  [1] GIF")
        print("  [2] MP4")
        fmt_choice = input("Enter choice (1/2) [1]: ").strip()
        save_format = "mp4" if fmt_choice == "2" else "gif"

        raw_name = input(
            "Enter filename WITHOUT extension (or press Enter for auto-generated): "
        ).strip()
        filename = Path(raw_name).stem if raw_name else None

    print("\n" + "-" * 70)
    print("Animating with:")
    print(f"  Mass (m):           {m} kg")
    print(f"  Stiffness (k):      {k} N/m")
    print(f"  Damping values:     {[float(np.round(c, 2)) for c in c_values if c > 0]}")
    print(f"  Initial disp (x0):  {x0} m")
    print(f"  Initial vel (v0):   {v0} m/s")
    print(f"  Periods:            {oscillations}")
    print(f"  FPS:                {fps}")
    if save_anim:
        print(f"  save_format:        {save_format}")
        print(f"  filename base:      {filename if filename else '(auto-generated)'}")
    print("-" * 70 + "\n")

    anim = animate_multiple_damping_cases_with_plots(
        m=m,
        c_values=c_values,
        k=k,
        x0=x0,
        v0=v0,
        oscillations=oscillations,
        fps=fps,
        save_animation=save_anim,
        filename=filename,
        save_format=save_format,
        forcing_func=forcing_func,
        forcing_name=forcing_name,
    )

    print("\nAnimation completed!")
    return anim


if __name__ == "__main__":
    animation = main()
