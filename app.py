"""Simple Harmonic Motion Studio.

The app is intentionally a thin orchestration layer over the existing physics
modules. This app calls the existing modules and displays the results in the form
of animations and plots.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from matplotlib.animation import HTMLWriter

import pendulum
import spring
import spring_resonance
from utils import (
    PALETTE,
    sinusoidal_forcing,
    no_forcing,
)


# ---------------------------------------------------------------------------
# Embed limit raised to prevent truncation on long animations
# ---------------------------------------------------------------------------
plt.rcParams["animation.embed_limit"] = 1000.0


# ---------------------------------------------------------------------------
# Theme loading
# ---------------------------------------------------------------------------
THEMES_DIR = Path(__file__).resolve().parent / "themes"

_DEFAULT_THEME: dict = {
    "name": "Deep Ocean (Default)",
    "gradient": {
        "background": "radial-gradient(circle at 8% 0%, #17313a 0%, #091217 34%, #060a0d 100%)",
        "accent_bar": "linear-gradient(90deg, #00EAFF, #AAFF00, #FF4D6D, #FF9F1C)",
    },
    "color": {
        "background": {
            "primary": "#091217",
            "secondary": "#101a20",
            "tertiary": "#14232b",
            "elevated": "#0c171d",
        },
        "text": {
            "primary": "#eef7f8",
            "secondary": "#b8c8cd",
            "muted": "#8ea0aa",
            "disabled": "#4a5a62",
            "on_accent": "#091217",
        },
        "border": {
            "default": "#21343c",
            "strong": "#28424c",
            "subtle": "#17282f",
        },
        "accent": {
            "primary": "#00EAFF",
            "secondary": "#AAFF00",
            "tertiary": "#FF4D6D",
            "quaternary": "#FF9F1C",
        },
        "semantic": {
            "success": "#AAFF00",
            "warning": "#FF9F1C",
            "error": "#FF4D6D",
            "info": "#00EAFF",
            "link": "#00EAFF",
        },
    },
    "typography": {
        "font_sans": "Inter, ui-sans-serif, system-ui, sans-serif",
        "font_mono": "'Space Mono', ui-monospace, monospace",
        "google_fonts_import": (
            "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800"
            "&family=Space+Mono:wght@400;700&display=swap"
        ),
    },
    "radius": {"sm": "8px", "md": "12px", "lg": "18px", "pill": "99px"},
    "shadow": {
        "glow_accent": "0 0 24px #00eaff44",
        "card": "0 4px 18px rgba(0,0,0,0.35)",
    },
}


def _load_themes() -> dict[str, dict]:
    """Load all JSON theme files from the themes/ directory plus the built-in default."""
    themes: dict[str, dict] = {"Deep Ocean (Default)": _DEFAULT_THEME}
    if THEMES_DIR.is_dir():
        for path in sorted(THEMES_DIR.glob("theme_*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                name = data.get("name", path.stem)
                themes[name] = data
            except Exception:
                pass
    return themes


THEMES: dict[str, dict] = _load_themes()


def inject_theme(theme: dict) -> None:
    """Inject a full CSS theme derived from the selected theme JSON."""
    bg = theme["color"]["background"]
    txt = theme["color"]["text"]
    acc = theme["color"]["accent"]
    bdr = theme["color"]["border"]
    grad = theme["gradient"]
    typ = theme["typography"]
    shd = theme["shadow"]
    rad = theme["radius"]

    st.markdown(
        f"""
        <style>
        @import url('{typ["google_fonts_import"]}');
        :root {{
            --bg:          {bg["primary"]};
            --panel:       {bg["secondary"]};
            --panel-2:     {bg["tertiary"]};
            --elevated:    {bg["elevated"]};
            --accent-1:    {acc["primary"]};
            --accent-2:    {acc["secondary"]};
            --accent-3:    {acc["tertiary"]};
            --accent-4:    {acc["quaternary"]};
            --text:        {txt["primary"]};
            --text-sec:    {txt["secondary"]};
            --muted:       {txt["muted"]};
            --border:      {bdr["default"]};
            --border-s:    {bdr["strong"]};
            --glow:        {shd["glow_accent"]};
            --rad-sm:      {rad["sm"]};
            --rad-md:      {rad["md"]};
            --rad-lg:      {rad["lg"]};
            --rad-pill:    {rad["pill"]};
            --font-sans:   {typ["font_sans"]};
            --font-mono:   {typ["font_mono"]};
        }}
        html, body, [data-testid="stAppViewContainer"] {{
            background: {grad["background"]};
            color: var(--text);
            font-family: var(--font-sans);
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        [data-testid="stSidebar"] {{
            background: linear-gradient(180deg, {bg["secondary"]} 0%, {bg["primary"]} 100%);
            border-right: 1px solid var(--border);
        }}
        [data-testid="stSidebar"] > div:first-child {{ padding-top: 1.25rem; }}
        .hero {{ padding: 0.8rem 0 0.3rem; }}
        .eyebrow {{
            color: var(--accent-1);
            font: 700 0.72rem var(--font-mono);
            letter-spacing: .18em;
            text-transform: uppercase;
        }}
        .hero h1 {{
            margin: .2rem 0 .25rem;
            font-size: clamp(2rem, 5vw, 4.2rem);
            letter-spacing: -.055em;
            line-height: 1;
        }}
        .hero p {{ color: var(--text-sec); max-width: 760px; font-size: 1.03rem; margin: 0; }}
        .spectrum {{
            height: 3px;
            margin: 1.2rem 0 1.4rem;
            border-radius: var(--rad-pill);
            background: {grad["accent_bar"]};
            box-shadow: var(--glow);
        }}
        .section-kicker {{
            color: var(--accent-1);
            font: 700 .75rem var(--font-mono);
            letter-spacing: .12em;
            text-transform: uppercase;
            margin: .35rem 0 .8rem;
        }}
        .theory-card, .empty-state {{
            background: linear-gradient(135deg, {bg["secondary"]}cc, {bg["tertiary"]}dd);
            border: 1px solid var(--border-s);
            border-radius: var(--rad-lg);
            padding: 1rem 1.15rem;
            margin-bottom: 1rem;
        }}
        .empty-state {{
            min-height: 155px;
            display: grid;
            place-items: center;
            text-align: center;
            color: var(--muted);
            border-style: dashed;
        }}
        .empty-orbit {{ color: var(--accent-1); font: 700 2.2rem var(--font-mono); letter-spacing: .15em; }}
        .caption {{ color: var(--muted); font-size: .86rem; }}
        .mono {{ font-family: var(--font-mono); }}
        div[data-testid="stMetric"] {{
            background: {bg["elevated"]};
            border: 1px solid var(--border);
            border-radius: var(--rad-md);
            padding: .8rem .9rem;
        }}
        div[data-testid="stMetricLabel"] {{ color: var(--muted); }}
        div[data-testid="stMetricValue"] {{ font-family: var(--font-mono); color: var(--text); }}
        .stButton > button, .stDownloadButton > button {{
            border-radius: var(--rad-md);
            border: 1px solid var(--border-s);
            background: linear-gradient(135deg, {bg["secondary"]}, {bg["tertiary"]});
            color: var(--text);
            font-weight: 700;
            transition: transform .15s, border-color .15s;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            transform: translateY(-1px);
            border-color: var(--accent-1);
            color: white;
        }}
        div[data-baseweb="tab-list"] {{ gap: .45rem; background: transparent; }}
        button[data-baseweb="tab"] {{
            color: var(--muted);
            border-radius: var(--rad-md) var(--rad-md) 0 0;
            padding: .75rem 1rem;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: white;
            background: {bg["secondary"]};
            box-shadow: inset 0 -3px 0 var(--accent-1);
        }}
        [data-testid="stExpander"] {{
            border-color: var(--border);
            background: {bg["elevated"]}99;
            border-radius: var(--rad-md);
        }}
        [data-testid="stSidebar"] label {{ color: var(--text-sec); }}
        footer {{ visibility: hidden; }}
        .footer {{
            border-top: 1px solid var(--border);
            margin-top: 2rem;
            padding: 1rem 0 0;
            color: var(--muted);
            font-size: .78rem;
            display: flex;
            justify-content: space-between;
            gap: 1rem;
            flex-wrap: wrap;
        }}
        @media (max-width: 768px) {{
            .hero h1 {{ font-size: 2.35rem; }}
            .hero p {{ font-size: .92rem; }}
            [data-testid="stSidebar"] {{ min-width: 300px; }}
            .footer {{ display: block; line-height: 1.7; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


def _set_default(key: str, value: Any) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------------------------------------------------------------------
# Preset button helper
# ---------------------------------------------------------------------------


def styled_preset_button(
    label: str,
    key: str,
    kwargs: dict,
    is_active: bool,
    active_color: str,
) -> None:
    marker_class = f"marker_{key}"
    st.markdown(
        f'<div class="{marker_class}" style="display:none;"></div>',
        unsafe_allow_html=True,
    )
    if is_active:
        st.markdown(
            f"""
        <style>
        div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button,
        div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button[kind="primary"],
        div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button[data-testid="stBaseButton-primary"],
        div.element-container:has(.{marker_class}) + div.element-container button {{
            background-color: {active_color} !important;
            background: {active_color} !important;
            border-color: {active_color} !important;
            color: white !important;
        }}
        div[data-testid="stElementContainer"]:has(.{marker_class}) + div[data-testid="stElementContainer"] button:hover,
        div.element-container:has(.{marker_class}) + div.element-container button:hover {{
            background-color: {active_color} !important;
            background: {active_color} !important;
            border-color: {active_color} !important;
            filter: brightness(1.15);
            color: white !important;
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )

    def _apply(**kw: Any) -> None:
        for k, v in kw.items():
            st.session_state[k] = v

    st.button(
        label,
        key=key,
        type="primary" if is_active else "secondary",
        use_container_width=True,
        on_click=_apply,
        kwargs=kwargs,
    )


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def common_controls() -> dict[str, Any]:
    _set_default("common_fps", 30)
    with st.sidebar.expander("\u2699\ufe0f Common Settings", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            g = st.number_input(
                "Gravity g (m/s\u00b2)",
                min_value=0.01,
                max_value=30.0,
                value=9.81,
                step=0.01,
                key="common_g",
                help="Gravitational acceleration in m/s\u00b2.",
            )
        with c2:
            fps = st.slider(
                "FPS",
                min_value=10,
                max_value=60,
                key="common_fps",
                help="Animation frames per second; higher values increase HTML size.",
            )
    return {"g": g, "fps": fps}


def sidebar_controls() -> dict[str, dict[str, Any]]:
    common = common_controls()

    # ── Simple Pendulum ───────────────────────────────────────────────────
    with st.sidebar.expander("\U0001f535 Simple Pendulum Parameters", expanded=True):
        _set_default("pend_theta0", 20.0)
        _set_default("pend_gamma1", 0.0)
        _set_default("pend_gamma2", 0.25)
        _set_default("pend_gamma3", 0.5)

        p1, p2, p3 = st.columns(3)
        cur_g1 = st.session_state.get("pend_gamma1", 0.0)
        cur_g2 = st.session_state.get("pend_gamma2", 0.25)
        cur_g3 = st.session_state.get("pend_gamma3", 0.5)
        with p1:
            styled_preset_button(
                "Undamped",
                "pend_preset_undamped",
                {"pend_gamma1": 0.0, "pend_gamma2": 0.0, "pend_gamma3": 0.0},
                cur_g1 == 0.0 and cur_g2 == 0.0 and cur_g3 == 0.0,
                "#00EAFF",
            )
        with p2:
            styled_preset_button(
                "Light Damp",
                "pend_preset_light",
                {"pend_gamma1": 0.0, "pend_gamma2": 0.25, "pend_gamma3": 0.5},
                cur_g1 == 0.0 and cur_g2 == 0.25 and cur_g3 == 0.5,
                "#AAFF00",
            )
        with p3:
            styled_preset_button(
                "Heavy Damp",
                "pend_preset_heavy",
                {"pend_gamma1": 0.0, "pend_gamma2": 0.5, "pend_gamma3": 1.0},
                cur_g1 == 0.0 and cur_g2 == 0.5 and cur_g3 == 1.0,
                "#FF4D6D",
            )

        c1, c2 = st.columns(2)
        with c1:
            theta0 = st.slider(
                "\u03b8\u2080 initial (\u00b0)",
                min_value=-30,
                max_value=30,
                value=20,
                key="pend_theta0",
                help="Initial angular displacement in degrees.",
            )
            L = st.number_input(
                "L (m)",
                min_value=0.01,
                value=1.0,
                step=0.1,
                key="pend_L",
                help="Pendulum length in metres.",
            )
            num_periods = st.slider(
                "Periods",
                min_value=1,
                max_value=20,
                value=5,
                key="pend_periods",
                help="Number of complete periods to simulate.",
            )
        with c2:
            gamma1 = st.number_input(
                "\u03b3\u2081 (s\u207b\u00b9)",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="pend_gamma1",
                help="Damping coefficient for pendulum 1 (0 = undamped).",
            )
            gamma2 = st.number_input(
                "\u03b3\u2082 (s\u207b\u00b9)",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="pend_gamma2",
                help="Damping coefficient for pendulum 2.",
            )
            gamma3 = st.number_input(
                "\u03b3\u2083 (s\u207b\u00b9)",
                min_value=0.0,
                max_value=5.0,
                step=0.05,
                key="pend_gamma3",
                help="Damping coefficient for pendulum 3.",
            )

    # ── Mass-Spring-Damper ────────────────────────────────────────────────
    with st.sidebar.expander("\U0001f527 Mass\u2013Spring\u2013Damper Parameters"):
        _set_default("msd_x0", 0.1)
        _set_default("msd_v0", 0.0)

        p1, p2, p3 = st.columns(3)
        with p1:
            styled_preset_button(
                "Undamped",
                "msd_preset_undamped",
                {"msd_c1": 0.0, "msd_c2": 0.5, "msd_c3": 6.32, "msd_c4": 8.0},
                False,
                "#00EAFF",
            )
        with p2:
            styled_preset_button(
                "Critical",
                "msd_preset_critical",
                {"msd_c1": 0.0, "msd_c2": 2.0, "msd_c3": 6.32, "msd_c4": 6.32},
                False,
                "#AAFF00",
            )
        with p3:
            styled_preset_button(
                "Overdamped",
                "msd_preset_over",
                {"msd_c1": 0.0, "msd_c2": 0.5, "msd_c3": 6.32, "msd_c4": 12.0},
                False,
                "#FF9F1C",
            )

        c1, c2 = st.columns(2)
        with c1:
            msd_m = st.number_input(
                "m (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="msd_m",
                help="Mass in kilograms.",
            )
            msd_k = st.number_input(
                "k (N/m)",
                min_value=0.001,
                value=10.0,
                step=0.5,
                key="msd_k",
                help="Spring stiffness in N/m.",
            )
            msd_x0 = st.number_input(
                "x\u2080 (m)",
                min_value=-5.0,
                max_value=5.0,
                value=0.1,
                step=0.05,
                key="msd_x0",
                help="Initial displacement from equilibrium.",
            )
            msd_v0 = st.number_input(
                "v\u2080 (m/s)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                key="msd_v0",
                help="Initial velocity.",
            )
        with c2:
            msd_c1 = st.number_input(
                "c\u2081 (N\u00b7s/m)",
                min_value=0.0,
                value=0.0,
                step=0.1,
                key="msd_c1",
                help="Damping coefficient case 1 (0 = undamped).",
            )
            msd_c2 = st.number_input(
                "c\u2082 (N\u00b7s/m)",
                min_value=0.0,
                value=0.5,
                step=0.1,
                key="msd_c2",
                help="Damping coefficient case 2 (underdamped).",
            )
            msd_c3 = st.number_input(
                "c\u2083 (N\u00b7s/m)",
                min_value=0.0,
                value=6.32,
                step=0.1,
                key="msd_c3",
                help="Damping coefficient case 3 (near-critical).",
            )
            msd_c4 = st.number_input(
                "c\u2084 (N\u00b7s/m)",
                min_value=0.0,
                value=8.0,
                step=0.1,
                key="msd_c4",
                help="Damping coefficient case 4 (overdamped).",
            )
        msd_osc = st.slider(
            "Oscillations",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=1.0,
            key="msd_osc",
            help="Number of natural periods to simulate.",
        )

    # ── Resonance ─────────────────────────────────────────────────────────
    with st.sidebar.expander("\U0001f30a Resonance Parameters"):
        _set_default("res_x0", 0.1)
        _set_default("res_v0", 0.0)

        p1, p2, p3 = st.columns(3)
        with p1:
            styled_preset_button(
                "Low Damp",
                "res_preset_low",
                {"res_c": 0.5, "res_F": 2.0},
                False,
                "#00EAFF",
            )
        with p2:
            styled_preset_button(
                "Mid Damp",
                "res_preset_mid",
                {"res_c": 2.0, "res_F": 2.0},
                False,
                "#AAFF00",
            )
        with p3:
            styled_preset_button(
                "High Damp",
                "res_preset_high",
                {"res_c": 4.0, "res_F": 2.0},
                False,
                "#FF4D6D",
            )

        c1, c2 = st.columns(2)
        with c1:
            res_m = st.number_input(
                "m (kg)",
                min_value=0.001,
                value=1.0,
                step=0.1,
                key="res_m",
                help="Mass in kilograms.",
            )
            res_k = st.number_input(
                "k (N/m)",
                min_value=0.001,
                value=10.0,
                step=0.5,
                key="res_k",
                help="Spring stiffness in N/m.",
            )
            res_c = st.number_input(
                "c (N\u00b7s/m)",
                min_value=0.0,
                value=2.0,
                step=0.1,
                key="res_c",
                help="Damping coefficient.",
            )
            res_F = st.number_input(
                "F\u2080 (N)",
                min_value=0.001,
                value=2.0,
                step=0.1,
                key="res_F",
                help="Forcing amplitude.",
            )
        with c2:
            res_x0 = st.number_input(
                "x\u2080 (m)",
                min_value=-5.0,
                max_value=5.0,
                value=0.1,
                step=0.05,
                key="res_x0",
                help="Initial displacement.",
            )
            res_v0 = st.number_input(
                "v\u2080 (m/s)",
                min_value=-20.0,
                max_value=20.0,
                value=0.0,
                step=0.1,
                key="res_v0",
                help="Initial velocity.",
            )
            res_osc = st.slider(
                "Oscillations",
                min_value=1.0,
                max_value=30.0,
                value=8.0,
                step=1.0,
                key="res_osc",
                help="Number of natural periods to simulate.",
            )

    return {
        "pendulum": {
            "L": L,
            "G": common["g"],
            "theta_0_deg": theta0,
            "gamma_list": [gamma1, gamma2, gamma3],
            "num_periods": num_periods,
            "fps": common["fps"],
        },
        "spring": {
            "m": msd_m,
            "c_values": [msd_c1, msd_c2, msd_c3, msd_c4],
            "k": msd_k,
            "x0": msd_x0,
            "v0": msd_v0,
            "oscillations": msd_osc,
            "fps": common["fps"],
        },
        "resonance": {
            "m": res_m,
            "c": res_c,
            "k": res_k,
            "x0": res_x0,
            "v0": res_v0,
            "forcing_amplitude": res_F,
            "oscillations": res_osc,
            "fps": common["fps"],
        },
    }


# ---------------------------------------------------------------------------
# Animation runners
# ---------------------------------------------------------------------------


def _run_pendulum(params: dict[str, Any]):
    return pendulum.animate_damped_vs_undamped_pendulum_with_plots(
        L=params["L"],
        G=params["G"],
        theta_0_deg=params["theta_0_deg"],
        gamma_list=params["gamma_list"],
        num_periods=params["num_periods"],
        fps=params["fps"],
        save_animation=False,
    )


def _run_spring(params: dict[str, Any]):
    return spring.animate_multiple_damping_cases_with_plots(
        m=params["m"],
        c_values=params["c_values"],
        k=params["k"],
        x0=params["x0"],
        v0=params["v0"],
        oscillations=params["oscillations"],
        fps=params["fps"],
        save_animation=False,
    )


def _build_resonance_args(params: dict[str, Any]) -> dict[str, Any]:
    """Derive forcing functions, labels, and computed frequencies from user params."""
    m = params["m"]
    k = params["k"]
    c = params["c"]
    F0 = params["forcing_amplitude"]

    omega_n = np.sqrt(k / m)
    f_n = omega_n / (2 * np.pi)
    c_critical = 2 * np.sqrt(k * m)
    zeta = c / c_critical

    if zeta < 1 / np.sqrt(2):
        omega_res = omega_n * np.sqrt(1 - 2 * zeta**2)
        f_ref = omega_res / (2 * np.pi)
        resonance_exists = True
    else:
        f_ref = f_n
        omega_res = omega_n
        resonance_exists = False

    f_low = f_ref * 0.5
    f_at = f_ref
    f_high = f_ref * 1.5

    c_values = [0, c, c, c]
    forcing_funcs = [
        no_forcing,
        lambda t, _fl=f_low: sinusoidal_forcing(t, F0, _fl),
        lambda t, _fa=f_at: sinusoidal_forcing(t, F0, _fa),
        lambda t, _fh=f_high: sinusoidal_forcing(t, F0, _fh),
    ]
    if resonance_exists:
        forcing_names = [
            "No forcing",
            f"f={f_low:.2f} Hz (< f_res)",
            f"f={f_at:.2f} Hz (= f_res)",
            f"f={f_high:.2f} Hz (> f_res)",
        ]
    else:
        forcing_names = [
            "No forcing",
            f"f={f_low:.2f} Hz (low)",
            f"f={f_at:.2f} Hz (mid)",
            f"f={f_high:.2f} Hz (high)",
        ]

    return {
        "c_values": c_values,
        "forcing_funcs": forcing_funcs,
        "forcing_names": forcing_names,
        "resonant_freq": f_ref if resonance_exists else None,
        "f_n": f_n,
        "f_ref": f_ref,
        "resonance_exists": resonance_exists,
        "zeta": zeta,
        "c_critical": c_critical,
        "omega_n": omega_n,
    }


def _run_resonance(params: dict[str, Any]):
    extra = _build_resonance_args(params)
    return spring_resonance.animate_damped_driven_oscillation_with_plots(
        m=params["m"],
        c_values=extra["c_values"],
        k=params["k"],
        x0=params["x0"],
        v0=params["v0"],
        oscillations=params["oscillations"],
        fps=params["fps"],
        save_animation=False,
        forcing_func=extra["forcing_funcs"],
        forcing_name=extra["forcing_names"],
        resonant_freq=extra["resonant_freq"],
    )


SYSTEMS: dict[str, dict[str, Any]] = {
    "pendulum": {
        "title": "\U0001f535 Simple Pendulum",
        "short": "Simple Pendulum",
        "accent": PALETTE["COLORS"][0],
        "runner": _run_pendulum,
        "default_filename": "pendulum",
        "description": "Undamped and damped pendulums oscillating from the same initial angle.",
    },
    "spring": {
        "title": "\U0001f527 Mass\u2013Spring\u2013Damper",
        "short": "Mass\u2013Spring\u2013Damper",
        "accent": PALETTE["COLORS"][1],
        "runner": _run_spring,
        "default_filename": "mass_spring_damper",
        "description": "Multiple damping regimes: undamped, underdamped, critically damped, overdamped.",
    },
    "resonance": {
        "title": "\U0001f30a Resonance",
        "short": "Resonance",
        "accent": PALETTE["COLORS"][2],
        "runner": _run_resonance,
        "default_filename": "mass_spring_resonance",
        "description": "Driven oscillator at, below, and above resonance frequency.",
    },
}


# ---------------------------------------------------------------------------
# Export / save animation
# ---------------------------------------------------------------------------


def _save_animation(
    system: str, params: dict[str, Any], fmt: str, filepath: Path
) -> None:
    """Render and save animation to filepath (gif or mp4)."""
    anim = SYSTEMS[system]["runner"](params)
    fmt = fmt.lower()
    if fmt == "mp4":
        from matplotlib.animation import FFMpegWriter

        writer = FFMpegWriter(
            fps=params["fps"],
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
        anim.save(str(filepath), writer=writer)
    else:
        anim.save(str(filepath), writer="ffmpeg", fps=params["fps"], dpi=80)
    fig = getattr(anim, "_fig", None)
    if fig is not None:
        plt.close(fig)


def export_panel(system: str, params: dict[str, Any]) -> None:
    st.markdown("<div class='section-kicker'>Export</div>", unsafe_allow_html=True)
    enabled = st.checkbox("Save this animation", key=f"export_enable_{system}")
    if not enabled:
        return
    fmt_label = st.radio(
        "Format", ["GIF", "MP4"], horizontal=True, key=f"export_format_{system}"
    )
    fmt = fmt_label.lower()
    default_name = SYSTEMS[system]["default_filename"]
    base = st.text_input(
        "Base filename",
        value=default_name,
        key=f"export_name_{system}",
        help="A safe base filename; the extension is added automatically.",
    )
    if st.button(
        "\U0001f4be Save Animation", key=f"save_{system}", use_container_width=True
    ):
        stem = Path(base.strip() or default_name).stem
        filename = f"{stem}.{fmt}"
        try:
            with st.spinner(f"Rendering {fmt.upper()} export..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp_path = Path(tmpdir) / filename
                    _save_animation(system, params, fmt, tmp_path)
                    if not tmp_path.exists():
                        raise RuntimeError(
                            "The animation writer did not create the expected file."
                        )
                    payload = tmp_path.read_bytes()
            st.session_state[f"download_{system}"] = (filename, payload, fmt)
            st.success(f"Generated {filename} successfully.")
        except Exception as exc:
            st.error(f"Could not save the animation: {exc}")
    download = st.session_state.get(f"download_{system}")
    if download:
        filename, payload, fmt = download
        st.download_button(
            "Download file",
            payload,
            file_name=filename,
            mime="video/mp4" if fmt == "mp4" else "image/gif",
            key=f"download_button_{system}",
            use_container_width=True,
        )


# ---------------------------------------------------------------------------
# Theory subsections
# ---------------------------------------------------------------------------


def theory_pendulum() -> None:
    with st.expander("THEORY \u00b7 equations, damping, and period", expanded=True):
        st.markdown(
            "A simple pendulum is governed by the damped nonlinear equation of motion. "
            "The damping term removes energy causing exponential amplitude decay. "
            "In the small-angle limit the solution is simple harmonic.\n\n"
            "**Variables:**\n"
            "- $\\theta$ \u2014 angular displacement (rad)\n"
            "- $L$ \u2014 pendulum length (m)\n"
            "- $g$ \u2014 gravitational acceleration (m/s\u00b2)\n"
            "- $\\gamma$ \u2014 damping coefficient (s\u207b\u00b9)\n"
            "- $\\omega_0 = \\sqrt{g/L}$ \u2014 natural angular frequency\n"
        )
        st.latex(r"\ddot{\theta} + \gamma\dot{\theta} + \frac{g}{L}\sin\theta = 0")
        col_a, col_b = st.columns(2, border=True)  # type: ignore
        with col_a:
            st.markdown("**Small-angle natural frequency and period:**")
            st.latex(
                r"\omega_0 = \sqrt{\frac{g}{L}}, \qquad T_0 = 2\pi\sqrt{\frac{L}{g}}"
            )
        with col_b:
            st.markdown("**Nonlinear period correction (elliptic integral):**")
            st.latex(
                r"T = T_0 \cdot \frac{2}{\pi} K\!\left(\sin^2\!\frac{\theta_0}{2}\right)"
            )


def theory_spring() -> None:
    with st.expander(
        "THEORY \u00b7 equations, damping regimes, and energy", expanded=True
    ):
        st.markdown(
            "A mass on a spring with viscous damping obeys the standard second-order ODE. "
            "Depending on the damping ratio the response is underdamped, critically damped, "
            "or overdamped.\n\n"
            "**Variables:**\n"
            "- $x$ \u2014 displacement from equilibrium (m)\n"
            "- $m$ \u2014 mass (kg)\n"
            "- $c$ \u2014 damping coefficient (N\u00b7s/m)\n"
            "- $k$ \u2014 spring stiffness (N/m)\n"
            "- $\\zeta = c/(2\\sqrt{mk})$ \u2014 damping ratio\n"
            "- $\\omega_n = \\sqrt{k/m}$ \u2014 natural angular frequency"
        )
        st.latex(r"m\ddot{x} + c\dot{x} + kx = 0")
        col_a, col_b = st.columns(2, border=True)  # type:ignore
        with col_a:
            st.markdown("**Natural frequency, damping ratio, and damped frequency:**")
            st.latex(
                r"\omega_n = \sqrt{\frac{k}{m}},\quad \zeta = \frac{c}{2\sqrt{mk}},\quad "
                r"\omega_d = \omega_n\sqrt{1-\zeta^2}"
            )
        with col_b:
            st.markdown("**Critical damping coefficient:**")
            st.latex(r"c_{\text{crit}} = 2\sqrt{mk}")


def theory_resonance() -> None:
    with st.expander("THEORY \u00b7 driven oscillator and resonance", expanded=True):
        st.markdown(
            "When a sinusoidal force drives the damped mass-spring system, the steady-state "
            "amplitude peaks near the resonant frequency.\n\n"
            "**Variables:**\n"
            "- $F_0$ \u2014 forcing amplitude (N) \n"
            "- $\\omega$ \u2014 driving angular frequency (rad/s) \n"
            "- $r = \\omega/\\omega_n$ \u2014 frequency ratio \n"
            "- $\\zeta$ \u2014 damping ratio"
        )
        st.latex(r"m\ddot{x} + c\dot{x} + kx = F_0\cos(\omega t)")
        col_a, col_b = st.columns(2, border=True)  # type:ignore
        with col_a:
            st.markdown("**Steady-state amplitude response:**")
            st.latex(
                r"X(\omega) = \frac{F_0/k}{\sqrt{\left(1-r^2\right)^2 + \left(2\zeta r\right)^2}}"
            )
        with col_b:
            st.markdown("**Amplitude-resonance frequency:**")
            st.latex(r"\omega_{\text{res}} = \omega_n\sqrt{1 - 2\zeta^2}")


# ---------------------------------------------------------------------------
# Property boxes
# ---------------------------------------------------------------------------

_BOX_CSS = """
<style>
.prop-row { display: flex; gap: 10px; margin: 10px 0 16px; flex-wrap: wrap; }
.prop-box {
    flex: 1 1 0;
    min-width: 140px;
    border-radius: 12px;
    padding: 12px 16px;
    border: 1px solid rgba(255,255,255,0.10);
    backdrop-filter: blur(4px);
}
.prop-box .pb-title {
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    opacity: 0.75;
    margin-bottom: 8px;
}
.prop-box .pb-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    gap: 4px;
    margin-bottom: 3px;
}
.prop-box .pb-label {
    font-size: 0.82rem;
    opacity: 0.82;
    white-space: nowrap;
    font-style: italic;
}
.prop-box .pb-label sub, .prop-box .pb-label sup {
    font-size: 0.70em;
    font-style: normal;
}
.prop-box .pb-label .pb-unit {
    font-size: 0.75rem;
    font-style: normal;
    opacity: 0.70;
}
.prop-box .pb-value {
    font-size: 0.90rem;
    font-weight: 700;
    font-family: 'Space Mono', monospace;
    white-space: nowrap;
}
</style>
"""


def _u(unit: str) -> str:
    """Wrap a unit string in a non-italic unit span."""
    return f"<span class='pb-unit'>&thinsp;{unit}</span>"


def _box_html(title: str, rows: list[tuple[str, str]], bg: str, accent: str) -> str:
    """Render a single property box as HTML."""
    row_html = "".join(
        f"<div class='pb-row'>"
        f"<span class='pb-label'>{lbl}</span>"
        f"<span class='pb-value' style='color:{accent}'>{val}</span>"
        f"</div>"
        for lbl, val in rows
    )
    return (
        f"<div class='prop-box' style='background:{bg};'>"
        f"<div class='pb-title' style='color:{accent}'>{title}</div>"
        f"{row_html}"
        f"</div>"
    )


def _property_boxes_pendulum(params: dict[str, Any]) -> None:
    """Render derived-property boxes for the simple pendulum."""
    from utils import calculate_nonlinear_period

    L = params["L"]
    G = params["G"]
    theta_0_deg = params["theta_0_deg"]
    gamma_list = params["gamma_list"]

    omega_0 = np.sqrt(G / L)
    T_small = 2 * np.pi * np.sqrt(L / G)
    T_nonlin = calculate_nonlinear_period(np.radians(theta_0_deg), L, G)
    f_0 = omega_0 / (2 * np.pi)

    def _damp_label(g: float) -> str:
        if g == 0:
            return "Undamped"
        if g < omega_0:
            return "Underdamped"
        if abs(g - omega_0) < 1e-6:
            return "Critically damped"
        return "Overdamped"

    box1 = _box_html(
        "\u03c9 &amp; f \u2014 Natural",
        [
            (f"\u03c9<sub>0</sub> {_u('(rad/s)')}", f"{omega_0:.4f}"),
            (f"f<sub>0</sub> {_u('(Hz)')}", f"{f_0:.4f}"),
        ],
        "rgba(0,234,255,0.08)",
        "#00EAFF",
    )
    box2 = _box_html(
        "Time Period",
        [
            (f"T<sub>0</sub> (small-angle) {_u('s')}", f"{T_small:.4f}"),
            (f"T (nonlinear) {_u('s')}", f"{T_nonlin:.4f}"),
            (
                f"\u0394T / T<sub>0</sub> {_u('(%)')}",
                f"{100 * (T_nonlin - T_small) / T_small:.3f}",
            ),
        ],
        "rgba(170,255,0,0.08)",
        "#AAFF00",
    )
    box3 = _box_html(
        "Damping Regimes",
        [
            (f"\u03b3<sub>1</sub> = {gamma_list[0]:.2f}", _damp_label(gamma_list[0])),
            (f"\u03b3<sub>2</sub> = {gamma_list[1]:.2f}", _damp_label(gamma_list[1])),
            (f"\u03b3<sub>3</sub> = {gamma_list[2]:.2f}", _damp_label(gamma_list[2])),
        ],
        "rgba(255,77,109,0.08)",
        "#FF4D6D",
    )

    st.markdown(_BOX_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='prop-row'>{box1}{box2}{box3}</div>",
        unsafe_allow_html=True,
    )


def _property_boxes_spring(params: dict[str, Any]) -> None:
    """Render derived-property boxes for the mass-spring-damper system."""
    m = params["m"]
    k = params["k"]
    c_values = params["c_values"]

    omega_n = np.sqrt(k / m)
    f_n = omega_n / (2 * np.pi)
    c_crit = 2 * np.sqrt(k * m)

    def _regime(c: float) -> str:
        zeta = c / c_crit
        if c == 0:
            return "Undamped"
        if abs(zeta - 1.0) < 0.02:
            return "Critical (\u03b6 \u2248 1)"
        if zeta < 1.0:
            return f"Under (\u03b6={zeta:.3f})"
        return f"Over (\u03b6={zeta:.3f})"

    def _ratio(c: float) -> str:
        ratio = c / c_crit
        if abs(ratio - 1.0) < 0.02:
            return "\u2248 1"
        return f"{ratio:.4f}"

    _u_nsm = _u("(N·s/m)")
    box1 = _box_html(
        "ω &amp; f — Natural",
        [
            (f"ω<sub>n</sub> {_u('(rad/s)')}", f"{omega_n:.4f}"),
            (f"f<sub>n</sub> {_u('(Hz)')}", f"{f_n:.4f}"),
            (f"T<sub>n</sub> {_u('(s)')}", f"{1 / f_n:.4f}"),
        ],
        "rgba(0,234,255,0.08)",
        "#00EAFF",
    )
    box2 = _box_html(
        "c<sub>crit</sub> &amp; ζ — Critical",
        [
            (f"c<sub>crit</sub> {_u_nsm}", f"{c_crit:.4f}"),
            (f"c<sub>1</sub> / c<sub>crit</sub>", _ratio(c_values[0])),
            (f"c<sub>2</sub> / c<sub>crit</sub>", _ratio(c_values[1])),
            (f"c<sub>3</sub> / c<sub>crit</sub>", _ratio(c_values[2])),
            (f"c<sub>4</sub> / c<sub>crit</sub>", _ratio(c_values[3])),
        ],
        "rgba(170,255,0,0.08)",
        "#AAFF00",
    )
    box3 = _box_html(
        "ζ — Damping Regimes",
        [
            (f"c<sub>1</sub> = {c_values[0]:.2f}", _regime(c_values[0])),
            (f"c<sub>2</sub> = {c_values[1]:.2f}", _regime(c_values[1])),
            (f"c<sub>3</sub> = {c_values[2]:.2f}", _regime(c_values[2])),
            (f"c<sub>4</sub> = {c_values[3]:.2f}", _regime(c_values[3])),
        ],
        "rgba(255,159,28,0.08)",
        "#FF9F1C",
    )

    st.markdown(_BOX_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='prop-row'>{box1}{box2}{box3}</div>",
        unsafe_allow_html=True,
    )


def _property_boxes_resonance(params: dict[str, Any]) -> None:
    """Render derived-property boxes for the resonance system."""
    extra = _build_resonance_args(params)

    omega_n = extra["omega_n"]
    f_n = extra["f_n"]
    c_critical = extra["c_critical"]
    zeta = extra["zeta"]
    c = params["c"]

    zeta_display = "\u2248 1" if abs(zeta - 1.0) < 0.02 else f"{zeta:.4f}"

    if extra["resonance_exists"]:
        omega_res = extra["f_ref"] * 2 * np.pi
        amp_ratio = (
            f"{1 / (2 * zeta * np.sqrt(1 - zeta**2)):.3f}" if zeta < 1 else "N/A"
        )
        res_row = [
            (f"ω<sub>res</sub> {_u('(rad/s)')}", f"{omega_res:.4f}"),
            (f"f<sub>res</sub> {_u('(Hz)')}", f"{extra['f_ref']:.4f}"),
            (f"X(ω<sub>res</sub>) / (F<sub>0</sub>/k)", amp_ratio),
        ]
        res_bg, res_accent = "rgba(191,90,242,0.08)", "#BF5AF2"
        res_title = "ω<sub>res</sub> — Resonance Peak"
    else:
        res_row = [
            ("Status", "No peak"),
            (f"ζ (current)", zeta_display),
            (f"ζ threshold (1/√2)", f"{1 / np.sqrt(2):.4f}"),
        ]
        res_bg, res_accent = "rgba(255,77,109,0.08)", "#FF4D6D"
        res_title = "ζ — Resonance Status"

    # Damping regime label
    if abs(zeta - 1.0) < 0.02:
        regime = "Critically damped"
    elif zeta < 1.0:
        regime = "Underdamped"
    else:
        regime = "Overdamped"

    _u_nsm = _u("(N·s/m)")
    box1 = _box_html(
        "ω &amp; f — Natural",
        [
            (f"ω<sub>n</sub> {_u('(rad/s)')}", f"{omega_n:.4f}"),
            (f"f<sub>n</sub> {_u('(Hz)')}", f"{f_n:.4f}"),
            (f"T<sub>n</sub> {_u('(s)')}", f"{1 / f_n:.4f}"),
        ],
        "rgba(0,234,255,0.08)",
        "#00EAFF",
    )
    box2 = _box_html(
        "c<sub>crit</sub> &amp; ζ — Damping",
        [
            (f"c<sub>crit</sub> {_u_nsm}", f"{c_critical:.4f}"),
            (f"ζ (= c / c<sub>crit</sub>)", zeta_display),
            (f"c {_u_nsm}", f"{c:.4f}"),
            ("Regime", regime),
        ],
        "rgba(170,255,0,0.08)",
        "#AAFF00",
    )
    box3 = _box_html(res_title, res_row, res_bg, res_accent)

    st.markdown(_BOX_CSS, unsafe_allow_html=True)
    st.markdown(
        f"<div class='prop-row'>{box1}{box2}{box3}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Simulation section
# ---------------------------------------------------------------------------


def simulation_section(system: str, params: dict[str, Any]) -> None:
    st.markdown(
        "<div class='section-kicker'>SIMULATION \u00b7 run, inspect, export</div>",
        unsafe_allow_html=True,
    )
    run_key = f"run_{system}"
    if st.button(
        "\u25b6 Run Simulation", key=run_key, type="primary", use_container_width=False
    ):
        try:
            start_time = time.time()
            with st.spinner("Solving equations of motion...", show_time=True):  # type: ignore
                anim = SYSTEMS[system]["runner"](params)
            solve_time = time.time() - start_time
            st.success(f"Equations solved in {solve_time:.2f} seconds!")

            progress_bar = st.progress(0.0, text="Generating animation frames...")

            def progress_cb(current_frame, total_frames):
                if total_frames:
                    progress = max(0.0, min(1.0, current_frame / total_frames))
                    progress_bar.progress(
                        progress,
                        text=(
                            f"Generating animation frames... "
                            f"({current_frame}/{total_frames} frames)"
                        ),
                    )
                else:
                    progress_bar.progress(
                        0.0,
                        text=f"Generating animation frames... (frame {current_frame})",
                    )

            fps = params.get("fps", 30)
            default_mode = "loop" if getattr(anim, "_repeat", False) else "once"

            anim_start = time.time()
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir, "temp.html")
                writer = HTMLWriter(
                    fps=fps, embed_frames=True, default_mode=default_mode
                )
                anim.save(str(path), writer=writer, progress_callback=progress_cb)
                html = path.read_text(encoding="utf-8")
            anim_time = time.time() - anim_start
            progress_bar.empty()
            st.success(f"Animation generated in {anim_time:.2f} seconds!")

            fig = getattr(anim, "_fig", None)
            if fig is not None:
                plt.close(fig)

            st.session_state[f"run_result_{system}"] = {"html": html, "params": params}
        except Exception as exc:
            st.error(f"Simulation could not run: {exc}")

    result = st.session_state.get(f"run_result_{system}")
    if result is None:
        st.markdown(
            "<div class='empty-state'><div>"
            "<div class='empty-orbit'>\u25cc  \u25cc  \u25cc</div>"
            "<div>Adjust the controls in the sidebar, then run the model to reveal its motion.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    # Height sized to fill ~90 % of a typical viewport
    components.html(result["html"], width="100%", height=900, scrolling=False)
    export_panel(system, result["params"])


# ---------------------------------------------------------------------------
# Render system tab
# ---------------------------------------------------------------------------


def render_system(system: str, params: dict[str, Any]) -> None:
    st.markdown(
        f"<h2 style='color:{SYSTEMS[system]['accent']}; margin-bottom:.15rem'>"
        f"{SYSTEMS[system]['short']}</h2>",
        unsafe_allow_html=True,
    )
    st.caption(SYSTEMS[system]["description"])

    # ── Theory subsection ─────────────────────────────────────────────────
    st.markdown(
        "<div class='section-kicker'>THEORY \u00b7 equations and key relations</div>",
        unsafe_allow_html=True,
    )
    if system == "pendulum":
        theory_pendulum()
        _property_boxes_pendulum(params)
    elif system == "spring":
        theory_spring()
        _property_boxes_spring(params)
    else:
        theory_resonance()
        _property_boxes_resonance(params)

    # ── Simulation subsection ─────────────────────────────────────────────
    simulation_section(system, params)


# ---------------------------------------------------------------------------
# App entry point
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Simple Harmonic Motion Demonstration",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main() -> None:
    with st.sidebar:
        st.markdown(
            "<div style='font-size:0.7rem; letter-spacing:.12em; text-transform:uppercase; "
            "color:#8ea0aa; margin-bottom:0.4rem; font-weight:700;'>"
            "\U0001f3a8 Theme</div>",
            unsafe_allow_html=True,
        )
        theme_names = list(THEMES.keys())
        selected_theme_name = st.selectbox(
            "Select theme",
            theme_names,
            index=0,
            key="selected_theme",
            label_visibility="collapsed",
        )
        selected_theme = THEMES[selected_theme_name]

    inject_theme(selected_theme)

    params_by_system = sidebar_controls()

    # ── Hero ──────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='hero'>"
        "<div class='eyebrow'>CLASSICAL MECHANICS \u00b7 SIMPLE HARMONIC MOTION</div>"
        "<h1>Simple Harmonic Motion Demonstration</h1>"
        "<p>Explore pendulum dynamics, damped oscillations, and resonance \u2014 "
        "all from a single interactive interface.</p>"
        "<div class='spectrum'></div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Tabs ──────────────────────────────────────────────────────────────
    tabs = st.tabs(
        [
            SYSTEMS["pendulum"]["title"],
            SYSTEMS["spring"]["title"],
            SYSTEMS["resonance"]["title"],
        ]
    )
    for tab, system in zip(tabs, ("pendulum", "spring", "resonance")):
        with tab:
            render_system(system, params_by_system[system])

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown(
        "<div class='footer'>"
        "<span>Built with Streamlit \u00b7 Matplotlib \u00b7 SciPy \u00b7 NumPy</span>"
        "<span>Simple Harmonic Motion Demonstration \u00b7 Physics Apps</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
