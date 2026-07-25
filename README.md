# ⏳ Simple Harmonic Motion — Interactive Physics Demonstration

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Live App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://simple-harmonic-motion-st-app-5r5akf8xdy8wkvxbodwji4.streamlit.app/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8%2B-11557c)](https://matplotlib.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.12%2B-8CAAE6)](https://scipy.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A premium, dark-themed **Streamlit** web application for interactively exploring the three canonical simple harmonic motion systems. Each simulation is rendered as a smooth, browser-native animation with live-updating plots, computed physics properties, and full export capability.

> 🌐 **Live Demo:** Explore the web app in your browser at [simple-harmonic-motion-st-app.streamlit.app](https://simple-harmonic-motion-st-app-5r5akf8xdy8wkvxbodwji4.streamlit.app/)

---

## ✨ Features at a Glance

| Feature | Details |
|---|---|
| 🎞 **Live Animations** | Smooth Matplotlib `FuncAnimation` rendered to interactive HTML5 in-browser players |
| 📐 **Three Simulations** | Damped Pendulum · Mass–Spring–Damper · Driven Resonance |
| 📊 **Real-time Plots** | Phase space, position/angle, velocity, and energy vs. time per simulation |
| 🧮 **Physics Property Boxes** | Colour-coded glassmorphism cards showing ω, f, T, ζ, c_crit, resonance peak etc. |
| 🎨 **Four Dark Themes** | Cyberpunk Slate · Midnight Emerald · Nebula Violet · Solar Graphite |
| 💾 **Export Animations** | Save to **GIF** or **MP4** directly from the sidebar |
| ⚙️ **Preset Configurations** | One-click parameter presets per system for instant exploration |
| 📖 **Inline Theory** | Expandable theory panels with equations typeset in LaTeX |

---

## 🖥️ Demo

### 🔵 Damped Pendulum

Multiple damping levels (undamped, underdamped, critically damped) animated simultaneously.  
Plots show angular position, angular velocity and phase-space trajectory.

> 📁 Sample animation:
    
https://github.com/user-attachments/assets/4a8dbbc4-4ef1-41b4-a2b4-d11cbc8fa3ca

---

### 🟢 Mass–Spring–Damper

Up to four spring-mass systems with individually configurable damping coefficients.  
Visualises position-time, phase space, and total mechanical energy.

> 📁 Sample animation:

https://github.com/user-attachments/assets/d607170d-3786-49e4-9a6b-20f9c3508f88

---

### 🟡 Driven Oscillator & Resonance

Sinusoidally forced mass–spring system at sub-resonant, resonant, and super-resonant driving frequencies.  
Demonstrates amplitude amplification and the resonance peak.

> 📁 Sample animation:


https://github.com/user-attachments/assets/e12d7fa4-1fca-4e06-baaa-c3d52d515d9f

---

## 🗂️ Project Structure

```
Simple Harmonic Motion/
│
├── app.py                    # Streamlit orchestration layer
├── pendulum.py               # Damped nonlinear pendulum physics + animation
├── spring.py                 # Mass–spring–damper physics + animation
├── spring_resonance.py       # Driven damped oscillator physics + animation
├── utils.py                  # Shared RK4 solvers, system analysis, colour palette
│
├── themes/
│   ├── theme_cyberpunk_slate.json
│   ├── theme_midnight_emerald.json
│   ├── theme_nebula_violet.json
│   └── theme_solar_graphite.json
│
└── outputs/                  # Sample exported animations
    ├── pendulum.mp4
    ├── mass_spring.mp4
    └── ms_resonance.mp4
```

---

## ⚙️ Physics Modules

### `pendulum.py` — Damped Nonlinear Pendulum

Solves the full nonlinear equation of motion using a 4th-order Runge–Kutta integrator:

$$
\ddot{\theta} + \gamma\dot{\theta} + \frac{g}{L}\sin\theta = 0
$$

- Compares undamped vs. multiple damping coefficients simultaneously
- Nonlinear period correction via elliptic integral
  
$$
T = T_0 \cdot \frac{2}{\pi} K\left(\sin^2\frac{\theta_0}{2}\right)
$$

- Live display of angular position, velocity, and phase-space portrait

---

### `spring.py` — Mass–Spring–Damper

Solves the standard damped free oscillation ODE:

$$m\ddot{x} + c\dot{x} + kx = 0$$

- Supports up to 4 simultaneous damping cases (undamped, underdamped, critically damped, overdamped)
- Auto-detects and labels damping regime using $\zeta = c / (2\sqrt{mk})$
- Plots position vs. time, phase space (v vs. x), and total mechanical energy

---

### `spring_resonance.py` — Driven Damped Oscillator

Simulates the steady-state forced response:

$$m\ddot{x} + c\dot{x} + kx = F_0\cos(\omega t)$$

| Driving case | Frequency |
|---|---|
| Sub-resonant | 0.5 $\omega_{res}$ |
| At resonance | $\omega_{res}$ |
| Super-resonant | 1.5 $\omega_{res}$ |


Resonance frequency, $\omega_{res} = \omega_{n} \sqrt{1 − 2\zeta^2}$ , valid only for $\zeta < 1/\sqrt{2}$.


---

### `utils.py` — Shared Utilities

| Function | Purpose |
|---|---|
| `rk4_mass_spring_damper` | RK4 solver for the mass–spring–damper ODE |
| `rk4_damped_pendulum` | RK4 solver for the nonlinear pendulum |
| `analyze_system_properties` | Computes $\omega_n$, $\zeta$, damping type, $\omega_d$ |
| `calculate_nonlinear_period` | Exact period via complete elliptic integral |
| `PALETTE` | Shared dark-theme colour constants |

---

## 🎨 Theming System

Themes are stored as structured JSON files in the `themes/` directory and injected at runtime.  
Each file defines a complete design token set:

```json
{
  "name": "Cyberpunk Slate",
  "color": {
    "background": { "primary": "#08090d", ... },
    "accent":     { "primary": "#ff2fb0", "secondary": "#29d3ff", ... }
  },
  "gradient": { "background": "...", "accent_bar": "..." },
  "typography": { "font_sans": "Inter", "font_mono": "Space Mono" }
}
```

| Theme | Description | Accents |
|---|---|---|
| **Cyberpunk Slate** | Near-black slate with electric neons | Magenta · Cyan · Yellow |
| **Midnight Emerald** | Deep navy with rich emerald highlights | Emerald · Teal · Amber |
| **Nebula Violet** | Dark indigo with violet and pink glows | Violet · Indigo · Pink |
| **Solar Graphite** | Warm charcoal with solar accents | Orange · Gold · Coral |

> 🖊️ **Adding a theme:** Drop a new `.json` following the same schema into `themes/` — it will be auto-detected on the next app launch.

---

## 🚀 Getting Started

### Prerequisites

- Python **3.10** or higher
- `pip` (or `uv` / `conda`)
- **FFmpeg** (optional — required only for MP4 export)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/PuspenduPH/simple-harmonic-motion-st-app.git
cd simple-harmonic-motion-st-app

# 2. Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Install Python dependencies
pip install streamlit matplotlib scipy numpy
```

For MP4 export, install [FFmpeg](https://ffmpeg.org/download.html) and ensure it is on your system `PATH`.

### Running the App

```bash
streamlit run app.py
```

The app opens automatically in your default browser at `http://localhost:8501`.

---

## 🕹️ Usage Guide

1. **Select a theme** from the sidebar's top drop-down.
2. **Choose a simulation tab** — Pendulum, Mass–Spring–Damper, or Resonance.
3. **Adjust parameters** in the sidebar sections (common settings, system-specific values, presets).
4. **Click ▶ Run Simulation** to generate the animation.
5. **Inspect the property boxes** below the theory expander — ω, f, T, ζ, c_crit, and regime labels update per run.
6. **Export** via the "Save Animation" checkbox in the sidebar; choose GIF or MP4 and click 💾 Save Animation.

> ⚠️ The animation is cached in `st.session_state`. After changing parameters, always click **Run Simulation** again to regenerate it.

---

## 📦 Dependencies

| Package | Minimum Version | Role |
|---|---|---|
| `streamlit` | 1.35 | Web framework and UI |
| `matplotlib` | 3.8 | Animation rendering and plots |
| `numpy` | 1.26 | Numerical arrays |
| `scipy` | 1.12 | Elliptic integrals (pendulum period) |
| `ffmpeg` *(optional)* | 6.0 | MP4 encoding |

---

## 📐 Key Equations Reference

| System | Governing Equation |
|---|---|
| **Pendulum** | $\ddot{\theta} + \gamma\dot{\theta} + \frac{g}{L}\sin\theta = 0$ |
| Small-angle frequency | $\omega_0 = \sqrt{g/L}$ , $T_0 = 2\pi\sqrt{L/g}$ |
| **Mass–Spring–Damper** | $m\ddot{x} + c\dot{x} + kx = 0$ |
| Natural frequency | $\omega_n = \sqrt{k/m}$ |
| Damping ratio | $\zeta = c\,/\,(2\sqrt{mk})$ |
| Critical damping | $c_\text{crit} = 2\sqrt{mk}$ |
| **Driven Oscillator** | $m\ddot{x} + c\dot{x} + kx = F_0\cos(\omega t)$ |
| Resonance frequency | $\omega_\text{res} = \omega_n\sqrt{1 - 2\zeta^2}$ |
| Steady-state amplitude | $X(\omega) = \dfrac{F_0/k}{\sqrt{(1-r^2)^2 + (2\zeta r)^2}}$ |

---

## 🤝 Contributing

Contributions, bug reports, and feature suggestions are welcome!  
Please open an issue or submit a pull request.

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

<p align="center">
  <sub>Built with ❤️ using Streamlit &nbsp;·&nbsp; Matplotlib &nbsp;·&nbsp; SciPy &nbsp;·&nbsp; NumPy</sub>
</p>
