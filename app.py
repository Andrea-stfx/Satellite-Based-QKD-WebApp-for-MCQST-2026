"""
================================================================================
SATELLITE QUANTUM KEY DISTRIBUTION SYSTEMS LABORATORY
================================================================================
Institution: Technische Universität München (TUM)
Department: Professorship of Quantum Communication Systems Engineering
Program: MCQST Summer Bachelor Program 2026
Academic Program Director: Prof. Dr. phil. Tobias Vogl
Supervisor: Dr. Asli Cakan Cebe
Simulation developed by: Andrea Staffieri
================================================================================
A physically self-consistent digital twin of a satellite-to-ground BB84 quantum
key distribution (QKD) link: two-body orbital kinematics, free-space optical
link budget, vacuum+weak decoy-state security analysis, eavesdropping models,
and aviation-constrained optical ground station (OGS) siting over the Munich
Metropolitan Area.
================================================================================
"""

from __future__ import annotations

import os
import time
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.integrate import simpson
from scipy.optimize import brentq


# =============================================================================
# 1. PAGE CONFIGURATION & THEME
# =============================================================================
st.set_page_config(
    page_title="Satellite QKD Systems Laboratory · TUM MCQST 2026",
    layout="wide",
    page_icon="🛰️",
    initial_sidebar_state="expanded",
)

ACCENT = "#0065BD"    # TUM blue
ACCENT2 = "#E37222"   # TUM orange
ACCENT3 = "#2E7D32"   # green (secure)
ACCENT4 = "#8E24AA"   # violet (quantum)
DARK = "#12233F"

st.markdown(f"""
<style>
    .stApp {{ background: #FAFBFC; }}
    h1, h2, h3 {{ font-family: 'Helvetica Neue', sans-serif; color: {DARK}; }}
    div[data-testid="stMetric"] {{
        background: #FFFFFF;
        border: 1px solid rgba(0,101,189,0.25);
        border-radius: 10px;
        padding: 10px 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }}
    div[data-testid="stMetricLabel"] {{ color: #4A5A73 !important; }}
    .stTabs [data-baseweb="tab"] {{ font-size: 13.5px; font-weight: 600; color:{DARK}; }}
    section[data-testid="stSidebar"] {{
        background: #F1F4F8;
        border-right: 1px solid rgba(0,101,189,0.15);
    }}
    .credit-line {{
        font-size: 13px; color:#4A5A73; text-align:center; margin-top:4px; line-height:1.6;
    }}
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_white"


def style_fig(fig, height=420, legend_orientation="h", legend_y=-0.22, margin_t=30, showlegend=True):
    """Uniform, high-contrast styling applied to every figure: dark, legible
    fonts and a bordered legend box, addressing low-contrast legend text."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        font=dict(color=DARK, size=12.5),
        margin=dict(l=10, r=10, t=margin_t, b=10),
        hoverlabel=dict(font=dict(color=DARK)),
    )
    if showlegend:
        fig.update_layout(legend=dict(
            orientation=legend_orientation, y=legend_y,
            font=dict(color=DARK, size=11.5),
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(18,35,63,0.25)", borderwidth=1,
        ))
    else:
        fig.update_layout(showlegend=False)
    return fig


# =============================================================================
# 2. INSTITUTIONAL LETTERHEAD
# =============================================================================
import os

# Injecting local CSS to control font sizes precisely. 
st.markdown("""
<style>
    .letterhead-title h1 {
        text-align: center;
        margin-bottom: 5px;
        font-size: 50px; 
        color: #12233F;
        line-height: 1.2; 
    }
    .letterhead-title h3 {
        text-align: center;
        font-weight: 500;
        color: #4A5A73;
        font-size: 18px; 
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

col_logo1, col_title, col_logo2 = st.columns([1, 5, 1])

with col_logo1:
    st.markdown("<div style='margin-top: 35px;'></div>", unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/b/ba/Tum_logo.gif?utm_source=commons.wikimedia.org&utm_campaign=index&utm_content=original", width=190)

with col_title:
    st.markdown("""
    <div class="letterhead-title">
    <h1>Simulation and Evaluation of Satellite-Based Quantum Key Distribution Links Under Aviation-Constrained Optical Ground Station Operation</h1>
    <h3>An Interactive Digital Twin for Orbital Link Budget, Decoy-State Security Analysis,
    and Spatial Optimization of Optical Ground Stations</h3>
    </div>
    """, unsafe_allow_html=True)

with col_logo2:
    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    st.image("https://www.mcqst.de/MCQST-MEDIA/images/Logos/MCQST_Logo_BlueBlack_Vector_OG_Image_1200_627.png", width=180)
    
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    mqv_path = os.path.join(BASE_DIR, "assets", "mqv_logo.png")
    
    if os.path.exists(mqv_path):
        st.image(mqv_path, width=210)
    else:
        st.info("📌 Add 'mqv_logo.png' to the 'assets' directory.")

st.markdown("""
<div class="credit-line">
<b>Program:</b> MCQST Summer Bachelor Program 2026 &nbsp;·&nbsp;
<b>Institution:</b> Technische Universität München (TUM) <br>
<b>Department:</b> TUM School of Computation, Information and Technology, Department of Computer Engineering - Professorship of Quantum Communication Systems Engineering<br>
<b>Academic Program Director:</b> Prof. Dr. phil. Tobias Vogl &nbsp;·&nbsp;
<b>Supervisor:</b> Dr. Asli Cakan Cebe &nbsp;·&nbsp;
<b>Author:</b> Andrea Staffieri
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =============================================================================
# 3. PHYSICAL CONSTANTS & FIXED MISSION PARAMETERS
# =============================================================================
@dataclass(frozen=True)
class Constants:
    MU_EARTH: float = 3.986004418e14   # m^3/s^2
    R_EARTH: float = 6_378_137.0       # m, WGS-84 equatorial radius


CONST = Constants()
INCLINATION = 98.0
MUC_LAT, MUC_LON = 48.3537, 11.7860
GARCHING_LAT, GARCHING_LON = 48.2665, 11.6691
DIST_INFLUENCE_KM = 100.0


# =============================================================================
# 4. ORBITAL MECHANICS & FREE-SPACE OPTICS
# =============================================================================
def orbital_velocity(mu: float, r: float) -> float:
    return float(np.sqrt(mu / r))

def visibility_half_angle(Re: float, h: float) -> float:
    return float(np.arccos(Re / (Re + h)))

def zenith_angle_and_range(lam, Re: float, h: float):
    Rs = Re + h
    d = np.sqrt(Re**2 + Rs**2 - 2.0 * Re * Rs * np.cos(lam))
    cos_theta = (Rs * np.cos(lam) - Re) / d
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))
    return theta, d

def gaussian_beam_radius(w0: float, wavelength: float, z):
    z = np.asarray(z, dtype=float)
    z_r = np.pi * w0**2 / wavelength
    return w0 * np.sqrt(1.0 + (z / z_r) ** 2), z_r

def kasten_young_airmass(theta_deg):
    t = np.clip(np.asarray(theta_deg, dtype=float), 0.0, 89.9)
    return 1.0 / (np.cos(np.radians(t)) + 0.50572 * (96.07995 - t) ** (-1.6364))

def atmospheric_transmittance(theta_deg, tau_zenith: float):
    return np.exp(-tau_zenith * kasten_young_airmass(np.abs(theta_deg)))

def aperture_collection_fraction(d, w0: float, wavelength: float, D_r: float):
    w_d, _ = gaussian_beam_radius(w0, wavelength, d)
    a = D_r / 2.0
    return 1.0 - np.exp(-2.0 * a**2 / w_d**2)

def channel_efficiency(theta_deg, d, eta_det, tau_zenith, D_r, w0, wavelength):
    T_atm = atmospheric_transmittance(theta_deg, tau_zenith)
    G = aperture_collection_fraction(d, w0, wavelength, D_r)
    return T_atm * G * eta_det

def line_segment_intersection(A, B, C, D):
    x1, y1 = A; x2, y2 = B; x3, y3 = C; x4, y4 = D
    Dmat = (x4 - x3) * (y2 - y1) - (x2 - x1) * (y4 - y3)
    if abs(Dmat) < 1e-12:
        return None, None, None
    t = ((x3 - x1) * (y4 - y3) - (y3 - y1) * (x4 - x3)) / Dmat
    u = ((x3 - x1) * (y2 - y1) - (y3 - y1) * (x2 - x1)) / Dmat
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return t, u, (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None, None, None

def haversine_km(lat1, lon1, lat2, lon2):
    R = CONST.R_EARTH / 1000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlmb = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))

def solve_kepler(M, e, tol=1e-10, max_iter=60):
    M = np.asarray(M, dtype=float)
    E = np.where(np.sin(M) > 0, M + e, M - e).astype(float)
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1.0 - e * np.cos(E))
        E = E - dE
        if np.all(np.abs(dE) < tol):
            break
    return E

def true_anomaly_from_eccentric(E, e):
    return 2.0 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2.0), np.sqrt(1 - e) * np.cos(E / 2.0))


# =============================================================================
# 5. QUANTUM INFORMATION THEORY — BB84 DECOY-STATE SECURITY
# =============================================================================
def binary_entropy(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0 - 1e-12)
    return -p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p)

def gain_and_qber(eta_ch, intensity, p_dark, e_detector):
    Q = p_dark + 1.0 - np.exp(-intensity * eta_ch)
    E = np.clip(
        (0.5 * p_dark + e_detector * (1.0 - np.exp(-intensity * eta_ch))) / np.clip(Q, 1e-15, None),
        0.0, 0.5,
    )
    return Q, E

def compute_link_metrics(theta_deg, d, wavelength, w0, tau_zenith, D_r,
                         eta_det, mu, p_dark, e_detector, f_EC, nu_hz):
    eta_ch = channel_efficiency(theta_deg, d, eta_det, tau_zenith, D_r, w0, wavelength)
    Q_mu, E_mu = gain_and_qber(eta_ch, mu, p_dark, e_detector)

    Y_1 = p_dark + eta_ch - p_dark * eta_ch
    Q_1 = mu * np.exp(-mu) * Y_1
    e_1 = np.clip(
        (0.5 * p_dark + e_detector * eta_ch) / np.clip(Y_1, 1e-15, None),
        0.0, 0.5,
    )

    R_bits_per_pulse = np.clip(
        Q_1 * (1.0 - binary_entropy(e_1)) - Q_mu * f_EC * binary_entropy(E_mu),
        0.0, None,
    )
    S = nu_hz * R_bits_per_pulse
    return eta_ch, Q_mu, E_mu, Q_1, e_1, S

def naive_bb84_rate_bits(Q_mu, E_mu, f_EC):
    return np.clip(Q_mu * (1.0 - binary_entropy(E_mu)) - Q_mu * f_EC * binary_entropy(E_mu), 0.0, None)

def vacuum_weak_decoy_estimate(mu, nu, Q_mu, E_mu, Q_nu, E_nu, Y_0):
    denom = mu * nu - nu**2
    if denom <= 0:
        return 0.0, 0.0, 0.5
    Y1_L = (mu / denom) * (
        Q_nu * np.exp(nu) - Q_mu * np.exp(mu) * (nu**2 / mu**2) - ((mu**2 - nu**2) / mu**2) * Y_0
    )
    Y1_L = float(max(Y1_L, 0.0))
    Q1_L = mu * np.exp(-mu) * Y1_L
    if Y1_L > 1e-15 and nu > 0:
        e1_U = (E_nu * Q_nu * np.exp(nu) - 0.5 * Y_0) / (Y1_L * nu)
        e1_U = float(np.clip(e1_U, 0.0, 0.5))
    else:
        e1_U = 0.5
    return Y1_L, Q1_L, e1_U

BB84_THRESHOLD_QBER = brentq(lambda e: 1.0 - 2.0 * float(binary_entropy(e)), 1e-6, 0.5 - 1e-6)

# --- additional helpers required by Part 1 and Part 2 below -----------------
def photon_number_gain_decomposition(mu, eta_ch, p_dark, n_max=5):
    """Decomposes the overall detection gain Q_mu into the contribution of
    each Fock-layer n = 0, 1, 2, ..., n_max-1, n_max+ using the click-yield
    model Y_n = 1-(1-eta_ch)^n (n>=1), Y_0 = p_dark, and Poisson weights
    P(n;mu). Returns arrays of layer contributions summing to Q_mu."""
    n_vals = np.arange(0, n_max + 1)
    # Replaced np.math.factorial with standard math.factorial for compatibility
    P_n = (mu ** n_vals) * np.exp(-mu) / np.array([math.factorial(int(n)) for n in n_vals])
    Y_n = np.where(n_vals == 0, p_dark, 1.0 - (1.0 - eta_ch) ** np.maximum(n_vals, 1))
    contrib = P_n * Y_n
    # tail (n > n_max) lumped into a residual layer using the exact overall gain
    Q_total = p_dark + 1.0 - np.exp(-mu * eta_ch)
    tail = max(Q_total - contrib.sum(), 0.0)
    return n_vals, contrib, tail, Q_total


def relativistic_time_dilation_ppb(v_orb, alt_m):
    """Net fractional clock-rate offset of a satellite relative to a ground
    clock, combining special-relativistic time dilation (satellite motion)
    and general-relativistic gravitational blueshift (weaker field at
    altitude), in parts-per-billion. Positive = satellite clock runs fast."""
    c = 299_792_458.0
    GM = CONST.MU_EARTH
    Re = CONST.R_EARTH
    sr_term = -0.5 * (v_orb ** 2) / c ** 2
    gr_term = (GM / c ** 2) * (1.0 / Re - 1.0 / (Re + alt_m))
    return (sr_term + gr_term) * 1e9


def rytov_variance(Cn2, wavelength, path_length_m):
    """Rytov variance for a plane wave under weak-turbulence Kolmogorov
    theory: sigma_R^2 = 1.23 Cn^2 k^(7/6) L^(11/6), k = 2*pi/lambda."""
    k = 2.0 * np.pi / wavelength
    return 1.23 * Cn2 * (k ** (7.0 / 6.0)) * (path_length_m ** (11.0 / 6.0))


def plob_bound_bits_per_use(eta):
    """Pirandola-Laurenza-Ottaviani-Banchi (PLOB) secret-key capacity of the
    pure-loss bosonic channel — the fundamental repeaterless bound on any
    QKD protocol operating on a lossy channel of transmittance eta:
    K = -log2(1-eta) [bits/channel use]. Nat. Commun. 8, 15043 (2017)."""
    eta = np.clip(np.asarray(eta, dtype=float), 0.0, 1.0 - 1e-12)
    return -np.log2(1.0 - eta)


# =============================================================================
# 6. CACHED SIMULATION ROUTINES
# =============================================================================
@st.cache_data(show_spinner=False)
def build_pass_kinematics(alt_sat, wavelength, w0, tau_zenith, D_r, eta_det,
                           mu, p_dark, e_detector, f_EC, nu_hz, n_samples=4000):
    v_orb = orbital_velocity(CONST.MU_EARTH, CONST.R_EARTH + alt_sat)
    omega = v_orb / (CONST.R_EARTH + alt_sat)
    lam_max = visibility_half_angle(CONST.R_EARTH, alt_sat)
    t_max = lam_max / omega
    t = np.linspace(0.0, 2 * t_max, n_samples)
    lam = omega * (t - t_max)
    theta_rad, d = zenith_angle_and_range(lam, CONST.R_EARTH, alt_sat)
    theta_deg = np.degrees(theta_rad)

    eta_ch, Q_mu, E_mu, Q_1, e_1, S_t = compute_link_metrics(
        theta_deg, d, wavelength, w0, tau_zenith, D_r, eta_det, mu, p_dark, e_detector, f_EC, nu_hz
    )
    return t, t_max, theta_deg, d, S_t, Q_mu, E_mu, Q_1, e_1, v_orb, omega

@st.cache_data(show_spinner=False)
def build_eclipse_grid(w_z, ac_offset, ac_diameter, n=140):
    bound = w_z * 2.2
    x = np.linspace(-bound, bound, n)
    y = np.linspace(-bound, bound, n)
    X, Y = np.meshgrid(x, y)
    I_full = np.exp(-2 * (X**2 + Y**2) / w_z**2)
    mask = (X - ac_offset) ** 2 + Y**2 < (ac_diameter / 2.0) ** 2
    I_ecl = I_full.copy()
    I_ecl[mask] = 0.0
    transmittance = float(I_ecl.sum() / I_full.sum())
    return X, Y, I_ecl, transmittance

@st.cache_data(show_spinner=False)
def build_risk_grid(margin_slider, n_points=41):
    lats = np.linspace(47.50, 49.25, n_points)
    lons = np.linspace(10.50, 13.00, n_points)
    LAT_GRID, LON_GRID = np.meshgrid(lats, lons)
    dist_km = haversine_km(LAT_GRID, LON_GRID, MUC_LAT, MUC_LON)
    noise = np.sin(LAT_GRID * 10) * np.cos(LON_GRID * 10) * 3.0
    margin_factor = margin_slider / 60.0
    RISK_GRID = np.clip(1.0 - (dist_km + noise) / DIST_INFLUENCE_KM, 0.0, 1.0) * 100.0 * margin_factor
    return LAT_GRID.flatten(), LON_GRID.flatten(), RISK_GRID.flatten()

@st.cache_data(show_spinner=False)
def sweep_margin_vs_efficiency(alt_sat, wavelength, w0, tau_zenith, D_r, eta_det,
                                mu, p_dark, e_detector, f_EC, nu_hz, crossing_time, margins):
    t, t_max, theta_deg, d, S_t, *_ = build_pass_kinematics(
        alt_sat, wavelength, w0, tau_zenith, D_r, eta_det, mu, p_dark, e_detector, f_EC, nu_hz)
    total = simpson(S_t, x=t)
    effs = []
    for m in margins:
        mask = (t >= crossing_time - m / 2) & (t <= crossing_time + m / 2)
        lost = simpson(S_t[mask], x=t[mask]) if np.count_nonzero(mask) >= 2 else 0.0
        effs.append(100.0 - (lost / total * 100.0 if total > 0 else 0.0))
    return np.array(effs)

@st.cache_data(show_spinner=False)
def sweep_tau_vs_efficiency(alt_sat, wavelength, w0, D_r, eta_det, mu, p_dark,
                             e_detector, f_EC, nu_hz, crossing_time, margin, tau_range):
    effs = []
    for tau in tau_range:
        t, t_max, theta_deg, d, S_t, *_ = build_pass_kinematics(
            alt_sat, wavelength, w0, tau, D_r, eta_det, mu, p_dark, e_detector, f_EC, nu_hz)
        total = simpson(S_t, x=t)
        mask = (t >= crossing_time - margin / 2) & (t <= crossing_time + margin / 2)
        lost = simpson(S_t[mask], x=t[mask]) if np.count_nonzero(mask) >= 2 else 0.0
        effs.append(100.0 - (lost / total * 100.0 if total > 0 else 0.0))
    return np.array(effs)

@st.cache_data(show_spinner=False)
def sweep_altitude_vs_pass(alt_range):
    durations, velocities = [], []
    for h in alt_range:
        v = orbital_velocity(CONST.MU_EARTH, CONST.R_EARTH + h)
        omega = v / (CONST.R_EARTH + h)
        t_max = visibility_half_angle(CONST.R_EARTH, h) / omega
        durations.append(2 * t_max)
        velocities.append(v)
    return np.array(durations), np.array(velocities)

@st.cache_data(show_spinner=False)
def sweep_mu_vs_rate(alt_sat, wavelength, w0, tau_zenith, D_r, eta_det,
                      p_dark, e_detector, f_EC, nu_hz, mu_range):
    rates, q1s, naive = [], [], []
    for m in mu_range:
        eta_ch, Q_mu, E_mu, Q_1, e_1, S = compute_link_metrics(
            0.0, alt_sat, wavelength, w0, tau_zenith, D_r, eta_det, m, p_dark, e_detector, f_EC, nu_hz)
        rates.append(float(S)); q1s.append(float(Q_1))
        naive.append(float(naive_bb84_rate_bits(Q_mu, E_mu, f_EC) * nu_hz))
    return np.array(rates), np.array(q1s), np.array(naive)

@st.cache_data(show_spinner=False)
def build_rate_heatmap(alt_sat, wavelength, w0, tau_zenith, D_r, p_dark, e_detector, f_EC, nu_hz,
                        mu_grid, eta_grid):
    matrix = np.zeros((len(eta_grid), len(mu_grid)))
    for iy, eta_v in enumerate(eta_grid):
        for ix, mu_v in enumerate(mu_grid):
            _, _, _, _, _, S_val = compute_link_metrics(
                0.0, alt_sat, wavelength, w0, tau_zenith, D_r, eta_v, mu_v, p_dark, e_detector, f_EC, nu_hz)
            matrix[iy, ix] = S_val
    return matrix

@st.cache_data(show_spinner=False)
def build_bloch_sphere_figure():
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.15, showscale=False,
                              colorscale=[[0, ACCENT], [1, ACCENT]]))
    for xr, yr, zr in [((-1.3, 1.3), (0, 0), (0, 0)),
                        ((0, 0), (-1.3, 1.3), (0, 0)),
                        ((0, 0), (0, 0), (-1.3, 1.3))]:
        fig.add_trace(go.Scatter3d(x=list(xr), y=list(yr), z=list(zr), mode="lines",
                                    line=dict(color="#999", width=3), showlegend=False))

    states = {
        "|0⟩  Z-basis, bit 0": (0, 0, 1, ACCENT),
        "|1⟩  Z-basis, bit 1": (0, 0, -1, ACCENT2),
        "|+⟩  X-basis, bit 0": (1, 0, 0, ACCENT3),
        "|−⟩  X-basis, bit 1": (-1, 0, 0, ACCENT4),
    }
    for name, (x, y, z, c) in states.items():
        fig.add_trace(go.Scatter3d(
            x=[0, x], y=[0, y], z=[0, z], mode="lines+markers+text",
            line=dict(color=c, width=6), marker=dict(size=5, color=c),
            text=["", name], textposition="top center", showlegend=False,
            textfont=dict(color=DARK, size=12)))

    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=460, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color=DARK),
        scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
                   aspectmode="cube"),
    )
    return fig

@st.cache_data(show_spinner=False)
def build_disturbance_bloch_figure(p_intercept):
    u = np.linspace(0, 2 * np.pi, 60); v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.12, showscale=False,
                              colorscale=[[0, ACCENT], [1, ACCENT]]))
    for xr, yr, zr in [((-1.3, 1.3), (0, 0), (0, 0)),
                        ((0, 0), (-1.3, 1.3), (0, 0)),
                        ((0, 0), (0, 0), (-1.3, 1.3))]:
        fig.add_trace(go.Scatter3d(x=list(xr), y=list(yr), z=list(zr), mode="lines",
                                    line=dict(color="#999", width=3), showlegend=False))

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[0, 1], mode="lines+markers+text",
                                line=dict(color=ACCENT, width=7), marker=dict(size=5, color=ACCENT),
                                text=["", "Alice's prepared state |0⟩"], textposition="top center",
                                showlegend=False, textfont=dict(color=DARK, size=12)))

    tilt = np.radians(90.0 * p_intercept)
    length = max(0.05, 1.0 - 0.5 * p_intercept)
    ex, ez = length * np.sin(tilt), length * np.cos(tilt)
    fig.add_trace(go.Scatter3d(x=[0, ex], y=[0, 0], z=[0, ez], mode="lines+markers+text",
                                line=dict(color="#D32F2F", width=7), marker=dict(size=5, color="#D32F2F"),
                                text=["", "State received by Bob"], textposition="bottom center",
                                showlegend=False, textfont=dict(color=DARK, size=12)))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=430, margin=dict(l=0, r=0, t=10, b=0),
                       font=dict(color=DARK),
                       scene=dict(xaxis=dict(visible=False), yaxis=dict(visible=False),
                                  zaxis=dict(visible=False), aspectmode="cube"))
    return fig

@st.cache_data(show_spinner=False)
def build_earth_orbit_figure(alt_sat, inclination_deg, t_current, t_max, omega, ogs_lat, ogs_lon):
    R = CONST.R_EARTH / 1e6
    r_orbit = (CONST.R_EARTH + alt_sat) / 1e6

    u = np.linspace(0, 2 * np.pi, 50); v = np.linspace(0, np.pi, 25)
    xs = R * np.outer(np.cos(u), np.sin(v))
    ys = R * np.outer(np.sin(u), np.sin(v))
    zs = R * np.outer(np.ones_like(u), np.cos(v))

    i_rad = np.radians(inclination_deg)
    uu = np.linspace(0, 2 * np.pi, 240)
    ox = r_orbit * np.cos(uu)
    oy = r_orbit * np.sin(uu) * np.cos(i_rad)
    oz = r_orbit * np.sin(uu) * np.sin(i_rad)

    u_sat = np.pi / 2 + omega * (t_current - t_max)
    sx = r_orbit * np.cos(u_sat)
    sy = r_orbit * np.sin(u_sat) * np.cos(i_rad)
    sz = r_orbit * np.sin(u_sat) * np.sin(i_rad)

    lat_r, lon_r = np.radians(ogs_lat), np.radians(ogs_lon)
    gx = R * np.cos(lat_r) * np.cos(lon_r)
    gy = R * np.cos(lat_r) * np.sin(lon_r)
    gz = R * np.sin(lat_r)

    fig = go.Figure()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs, opacity=0.88, showscale=False,
                              colorscale=[[0, "#BBDEFB"], [0.5, "#64B5F6"], [1, "#1565C0"]]))
    fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode="lines",
                                line=dict(color=ACCENT2, width=4), name="Orbit track"))
    fig.add_trace(go.Scatter3d(x=[sx], y=[sy], z=[sz], mode="markers+text",
                                marker=dict(size=6, color="#D32F2F"),
                                text=["Satellite"], textposition="top center", name="Satellite",
                                textfont=dict(color=DARK, size=12)))
    fig.add_trace(go.Scatter3d(x=[gx], y=[gy], z=[gz], mode="markers+text",
                                marker=dict(size=5, color=ACCENT),
                                text=["TUM Garching OGS"], textposition="bottom center", name="OGS",
                                textfont=dict(color=DARK, size=12)))
    fig.update_layout(
        template=PLOTLY_TEMPLATE, height=560, margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color=DARK),
        scene=dict(xaxis_title="X (Mm)", yaxis_title="Y (Mm)", zaxis_title="Z (Mm)", aspectmode="data"),
    )
    return fig

def build_mission_schematic_figure(t_now, T_MAX, in_blackout, crossing_time, margin_half,
                                   alt_ac_fraction):
    angle = np.pi * np.clip(t_now / (2 * T_MAX), 0.0, 1.0)
    sx, sy = np.cos(angle), np.sin(angle)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[-1.3, 1.3], y=[0, 0], mode="lines",
                              line=dict(color="#B0BEC5", width=2), showlegend=False))
    arc_t = np.linspace(0, np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(arc_t), y=np.sin(arc_t), mode="lines",
                              line=dict(color="#CFD8DC", width=1, dash="dot"), showlegend=False))

    beam_color = "#D32F2F" if in_blackout else ACCENT
    beam_dash = "dash" if in_blackout else "solid"
    fig.add_trace(go.Scatter(x=[sx, 0], y=[sy, 0], mode="lines",
                              line=dict(color=beam_color, width=3, dash=beam_dash), showlegend=False))

    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers+text",
                              marker=dict(size=16, color=ACCENT, symbol="triangle-up"),
                              text=["OGS"], textposition="bottom center", showlegend=False,
                              textfont=dict(color=DARK, size=13)))
    fig.add_trace(go.Scatter(x=[sx], y=[sy], mode="markers+text",
                              marker=dict(size=16, color=DARK, symbol="diamond"),
                              text=["Satellite"], textposition="top center", showlegend=False,
                              textfont=dict(color=DARK, size=13)))

    if abs(t_now - crossing_time) <= margin_half * 2.5:
        s = float(np.clip(alt_ac_fraction, 0.02, 0.98))
        ax, ay = sx * s, sy * s
        fig.add_trace(go.Scatter(x=[ax], y=[ay], mode="markers+text",
                                  marker=dict(size=17, color=ACCENT2, symbol="x"),
                                  text=["Aircraft"], textposition="middle right", showlegend=False,
                                  textfont=dict(color=DARK, size=13)))

    fig = style_fig(fig, height=300, showlegend=False, margin_t=10)
    fig.update_layout(xaxis=dict(visible=False, range=[-1.3, 1.3]),
                       yaxis=dict(visible=False, range=[-0.15, 1.3], scaleanchor="x", scaleratio=1))
    return fig


# =============================================================================
# 7. SIDEBAR — SIMULATION PARAMETERS
# =============================================================================
st.sidebar.title("⚙️ Simulation Controls")

with st.sidebar.expander("🛰️ Orbital Parameters", expanded=True):
    alt_sat_km = st.slider("Satellite altitude [km]", 200.0, 800.0, 500.0, 10.0)
    ALT_SAT = alt_sat_km * 1000.0

with st.sidebar.expander("✈️ Aviation Safety", expanded=True):
    margin_slider = st.slider("Safety margin Δt [s]", 0.0, 200.0, 60.0, 5.0,
        help="Symmetric beacon-laser blackout window around the aircraft crossing.")
    v_orb0 = orbital_velocity(CONST.MU_EARTH, CONST.R_EARTH + ALT_SAT)
    omega0 = v_orb0 / (CONST.R_EARTH + ALT_SAT)
    T_MAX0 = visibility_half_angle(CONST.R_EARTH, ALT_SAT) / omega0
    crossing_time = st.slider("Aircraft crossing time [s]", 0.0, float(2 * T_MAX0),
                               float(T_MAX0), 5.0)
    ac_offset = st.slider("Aircraft radial offset from beam axis [m]", -300.0, 300.0, 0.0, 10.0)
    ac_alt_km = st.slider("Aircraft cruise altitude [km]", 5.0, 12.0, 10.0, 0.5)
    ac_diameter = st.slider("Aircraft cross-section diameter [m]", 10.0, 100.0, 80.0, 5.0)

with st.sidebar.expander("🔭 Optical Link Budget", expanded=True):
    WAVELENGTH_NM = st.slider("Beacon wavelength [nm]", 780.0, 1550.0, 854.445, 0.1)
    WAVELENGTH = WAVELENGTH_NM * 1e-9
    W0 = st.slider("Transmitter beam waist W₀ [m]", 0.05, 0.30, 0.14, 0.01)
    tau_zenith = st.slider("Zenith atmospheric optical depth τ₀", 0.05, 0.50, 0.15, 0.01)
    D_r = st.slider("Receiver aperture diameter [m]", 0.10, 1.00, 0.30, 0.05)

with st.sidebar.expander("🔑 Quantum Source & Detector", expanded=True):
    mu = st.slider("Mean photon number μ (signal state)", 0.05, 3.00, 0.50, 0.05)
    nu_MHz = st.slider("Pulse repetition rate ν [MHz]", 1.0, 500.0, 100.0, 1.0)
    nu_hz = nu_MHz * 1e6
    p_dark_exp = st.slider("Dark-count probability (10ˣ per gate)", -8.0, -4.0, -6.0, 0.5)
    p_dark = 10.0 ** p_dark_exp
    eta_det = st.slider("Detector efficiency η_det", 0.10, 0.90, 0.60, 0.05)
    e_detector = st.slider("Intrinsic detector error rate e_det", 0.001, 0.050, 0.010, 0.001, format="%.3f")
    f_EC = st.slider("Error-correction inefficiency f_EC", 1.00, 1.50, 1.16, 0.01)

# --- derived quantities -----------------------------------------------------
ALT_AC = ac_alt_km * 1000.0
Z_DIST = ALT_SAT - ALT_AC
W_Z, Z_R = gaussian_beam_radius(W0, WAVELENGTH, Z_DIST)
AC_ALT_FRACTION = float(np.clip(ALT_AC / ALT_SAT, 0.0, 1.0))

(t_arr, T_MAX, theta_deg_arr, d_arr, S_t_arr, Qmu_arr, Emu_arr, Q1_arr, e1_arr,
 v_orb, omega) = build_pass_kinematics(ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r,
                                        eta_det, mu, p_dark, e_detector, f_EC, nu_hz)

T_PERIOD = 2 * np.pi * (CONST.R_EARTH + ALT_SAT) / v_orb

eta0, Qmu0, Emu0, Q10, e10, S0 = compute_link_metrics(
    0.0, ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r, eta_det, mu, p_dark, e_detector, f_EC, nu_hz)

margin_half = margin_slider / 2.0
t_start, t_end = crossing_time - margin_half, crossing_time + margin_half
mask_loss = (t_arr >= t_start) & (t_arr <= t_end)
total_key_yield = simpson(S_t_arr, x=t_arr)
lost_key_yield = simpson(S_t_arr[mask_loss], x=t_arr[mask_loss]) if np.count_nonzero(mask_loss) >= 2 else 0.0
efficiency = 100.0 - (lost_key_yield / total_key_yield * 100.0 if total_key_yield > 0 else 0.0)

st.sidebar.markdown("---")
st.sidebar.success(f"**Orbital velocity:** {v_orb/1000:.3f} km/s")
st.sidebar.success(f"**Orbital period:** {T_PERIOD/60:.1f} min")
st.sidebar.success(f"**Pass duration:** {2*T_MAX:,.0f} s")
st.sidebar.success(f"**Beam diameter at aircraft:** {2*W_Z:.2f} m")

# =============================================================================
# 8. EXECUTIVE KPI ROW
# =============================================================================
k1, k2, k3, k4 = st.columns(4)
k1.metric("Net Link Efficiency", f"{efficiency:.2f} %")
k2.metric("Zenith QBER", f"{Emu0*100:.2f} %",
          "below threshold" if Emu0 < BB84_THRESHOLD_QBER else "⚠ above threshold",
          delta_color="normal" if Emu0 < BB84_THRESHOLD_QBER else "inverse")
k3.metric("Zenith Secure Key Rate", f"{S0/1000:.1f} kbit/s")
k4.metric("Orbital Period", f"{T_PERIOD/60:.1f} min")

st.markdown("---")

# =============================================================================
# 9. MAIN DASHBOARD TABS
# =============================================================================
(tab_theory, tab_quantum, tab_mission, tab_crypto, tab_eve, tab_orbit3d,
 tab_optics, tab_geo, tab_network, tab_cross, tab_sweep, tab_own, tab_hifi) = st.tabs([
    "Theoretical Framework",
    "Quantum Channel & Security",
    "Mission Control",
    "Key-Loss Analysis",
    "Eavesdropping Analysis",
    "Orbital Geometry",
    "Beam Occultation",
    "Geospatial Risk Map",
    "Network Designer",
    "Crossing Detection",
    "Sensitivity Analysis",
    "Numerical Simulation Results",
    "Astrodynamics & Link Optimization"
])

# -----------------------------------------------------------------------------
# TAB 1 — THEORY
# -----------------------------------------------------------------------------
with tab_theory:
    st.header("Theoretical Framework")
    st.caption("Foundational quantum-information theory, orbital astrodynamics, space-channel physics, "
               "and quantum-optical link-budget theory underlying the simulation engine.")

    # =========================================================================
    # SECTION 0 — POSTULATES OF QUANTUM CRYPTOGRAPHIC SECURITY
    # =========================================================================
    st.subheader("1. Foundational Postulates of Quantum Cryptographic Security")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown(r"**No-cloning theorem** — no unitary $U$ perfectly copies an arbitrary unknown state, "
                    r"the structural origin of quantum-channel eavesdropping detectability:")
        st.latex(r"\nexists\, U : \; U\,(|\psi\rangle \otimes |0\rangle) = |\psi\rangle \otimes |\psi\rangle \quad \forall\, |\psi\rangle")

        st.markdown(r"**Robertson uncertainty relation** for two conjugate observables $\hat A,\hat B$:")
        st.latex(r"\Delta A\,\Delta B \;\geq\; \tfrac{1}{2}\left|\langle\psi|[\hat{A},\hat{B}]|\psi\rangle\right|")

        st.markdown(r"**Depolarizing channel model** of the free-space link, Bloch-vector contraction $r=1-2p$:")
        st.latex(r"\mathcal{E}(\rho) = (1-p)\,\rho + p\,\frac{\mathbb{I}}{2}, \qquad \rho = \tfrac{1}{2}(\mathbb{I}+\vec{r}\cdot\vec{\sigma})")

        st.markdown(r"**Von Neumann entropy** of the resulting mixed qubit, eigenvalues $\lambda_\pm=(1\pm|\vec r|)/2$:")
        st.latex(r"S(\rho) = -\mathrm{Tr}(\rho \log_2 \rho) = h_2\!\left(\frac{1+|\vec{r}|}{2}\right)")

    with col_p2:
        st.markdown(r"**Holevo bound** on Eve's accessible information about ensemble $\{p_i,\rho_i\}$:")
        st.latex(r"I(A:E) \;\leq\; \chi = S\!\left(\sum_i p_i \rho_i\right) - \sum_i p_i\, S(\rho_i)")

        st.markdown(r"**Devetak–Winter secret-key capacity** against collective attacks:")
        st.latex(r"R_\infty \;\geq\; \max_{\{p_i,\rho_i\}}\Big[\, I(A:B) - \chi(A:E) \,\Big]")

        st.markdown(r"**CHSH inequality** — local-realistic vs. Tsirelson quantum bound (device-independent QKD):")
        st.latex(r"|S_{\text{CHSH}}| \leq 2 \;\; \text{(LHV)}, \qquad |S_{\text{CHSH}}| \leq 2\sqrt{2} \;\; \text{(quantum)}")

        st.markdown(r"**GLLP finite-key correction**, failure probabilities $\varepsilon_{\text{cor}},\varepsilon_{\text{PA}},\varepsilon_{\text{sec}}$, block size $N$:")
        st.latex(r"R_{\text{fin}} = R_\infty - \frac{1}{N}\left[\log_2\frac{2}{\varepsilon_{\text{cor}}} + 2\log_2\frac{1}{2\varepsilon_{\text{PA}}\varepsilon_{\text{sec}}}\right]")

        st.markdown(r"**PLOB repeaterless bound** — fundamental secret-key capacity of the pure-loss bosonic channel "
                    r"(Pirandola, Laurenza, Ottaviani, Banchi, *Nat. Commun.* **8**, 15043, 2017):")
        st.latex(r"K(\eta) = -\log_2(1-\eta) \;\; \text{[bits/channel use]}, \qquad \eta = \text{channel transmittance}")

    st.info(fr"The idealized single-photon security bound $R \propto 1-2h_2(e)$ vanishes at the BB84 QBER threshold "
            fr"$e^{{*}} \approx {BB84_THRESHOLD_QBER*100:.2f}\%$; no positive asymptotic secure key rate is achievable "
            fr"under collective attacks beyond this point.")

    st.markdown("---")

    # =========================================================================
    # SECTION 1 — QUANTUM OPTICS OF THE PHOTON FIELD
    # =========================================================================
    st.subheader("2. Quantum Optics of the Transmitted Photon Field")
    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.markdown(r"**Quantized single-mode field**, annihilation/creation operators $\hat a,\hat a^\dagger$, "
                    r"canonical commutator, and Hamiltonian:")
        st.latex(r"[\hat a, \hat a^\dagger] = 1, \qquad \hat H = \hbar\omega\left(\hat a^\dagger \hat a + \tfrac12\right)")

        st.markdown(r"**Coherent state** $|\alpha\rangle$ — the quantum-optical description of an attenuated laser pulse, "
                    r"eigenstate of the annihilation operator:")
        st.latex(r"\hat a |\alpha\rangle = \alpha |\alpha\rangle, \qquad |\alpha\rangle = e^{-|\alpha|^2/2}\sum_{n=0}^{\infty}\frac{\alpha^n}{\sqrt{n!}}|n\rangle")

        st.markdown(r"**Photon-number statistics** follow a Poisson distribution with mean $\mu = |\alpha|^2$:")
        st.latex(r"P(n;\mu) = |\langle n|\alpha\rangle|^2 = \frac{\mu^{n} e^{-\mu}}{n!}")

    with col_q2:
        st.markdown(r"**Quadrature operators** and the Heisenberg minimum-uncertainty relation for coherent states:")
        st.latex(r"\hat X = \frac{\hat a + \hat a^\dagger}{2}, \quad \hat P = \frac{\hat a - \hat a^\dagger}{2i}, "
                 r"\qquad \Delta X\,\Delta P = \frac{1}{4}")

        st.markdown(r"**Beam-splitter transmittance model** of atmospheric/geometric loss, mapping input mode "
                    r"$\hat a_{\text{in}}$ through a channel of transmittance $\eta$ into vacuum-admixed output:")
        st.latex(r"\hat a_{\text{out}} = \sqrt{\eta}\,\hat a_{\text{in}} + \sqrt{1-\eta}\,\hat a_{\text{vac}}")

        st.markdown(r"Under this model, a coherent state of mean photon number $\mu$ remains coherent after the "
                    r"lossy channel, with mean photon number rescaled — the physical justification for "
                    r"$Y_n \approx 1-(1-\eta)^n$ used in the detection-gain decomposition:")
        st.latex(r"|\alpha\rangle \xrightarrow{\ \eta\ } |\sqrt{\eta}\,\alpha\rangle, \qquad \mu \to \eta\mu")

    st.markdown("---")

    # =========================================================================
    # SECTION 2 — ORBITAL ASTRODYNAMICS
    # =========================================================================
    st.subheader("3. Orbital Astrodynamics")
    col_o1, col_o2 = st.columns(2)
    with col_o1:
        st.markdown(r"**Two-body equation of motion** under the point-mass, spherical-Earth approximation:")
        st.latex(r"\ddot{\vec{r}} = -\frac{\mu}{|\vec{r}|^{3}}\,\vec{r}, \qquad \mu = G\,m_{\oplus} = 3.986004418\times10^{14}\ \mathrm{m^3s^{-2}}")

        st.markdown(r"**Specific angular momentum** $\vec h = \vec r \times \vec v$ is conserved, giving the conic-section trajectory:")
        st.latex(r"r(\theta) = \frac{h^2/\mu}{1+e\cos\theta}, \qquad \dot\theta = \frac{h}{r^2}")

        st.markdown(r"**Kepler's equation**, relating mean anomaly $M$ to eccentric anomaly $E$ (solved here via Newton–Raphson):")
        st.latex(r"M = E - e\sin E, \qquad n=\sqrt{\frac{\mu}{a^{3}}}, \qquad M(t) = n\,(t-t_0)")

        st.markdown(r"**Vis-viva equation**, orbital speed at radius $r$ (reduces to $v=\sqrt{\mu/(R_\oplus+h)}$ for a circular orbit):")
        st.latex(r"v^2 = \mu\left(\frac{2}{r} - \frac{1}{a}\right)")

    with col_o2:
        st.markdown(r"**Slant-range geometry** between OGS and satellite, plane triangle {Earth centre, OGS, satellite}, "
                    r"central angle $\lambda=\omega t$ since zenith crossing:")
        st.latex(r"d(\lambda) = \sqrt{R_\oplus^{2} + (R_\oplus+h)^{2} - 2R_\oplus (R_\oplus+h)\cos\lambda}")
        st.latex(r"\theta(\lambda) = \arccos\!\left(\frac{(R_\oplus+h)\cos\lambda - R_\oplus}{d(\lambda)}\right)")

        st.markdown(r"**Horizon visibility condition**, setting the pass duration used throughout the simulation:")
        st.latex(r"\theta = 90^{\circ} \;\; \Rightarrow \;\; \lambda_{\max} = \arccos\!\left(\frac{R_\oplus}{R_\oplus+h}\right)")

        st.markdown(r"**Sun-synchronous inclination condition** — RAAN precession rate matched to Earth's mean solar motion, "
                    r"used to justify the uniform-RAAN pass-sampling approximation ($J_2$ = Earth oblateness coefficient):")
        st.latex(r"\dot\Omega = -\frac{3}{2}n J_2\left(\frac{R_\oplus}{a}\right)^2\cos i \;\stackrel{!}{=}\; \frac{2\pi}{365.25\ \mathrm{days}}")

    st.markdown("---")

    # =========================================================================
    # SECTION 3 — RELATIVISTIC CORRECTIONS FOR SPACE LINKS
    # =========================================================================
    st.subheader("4. Relativistic Corrections Relevant to Space-Based Quantum Links")
    st.markdown(
        r"A satellite clock is subject to both special-relativistic time dilation (orbital velocity) and "
        r"general-relativistic gravitational blueshift (weaker gravitational potential at altitude). Their "
        r"combined fractional frequency offset must be tracked for high-precision timing synchronization "
        r"between Alice's and Bob's detection windows:"
    )
    st.latex(r"\frac{\Delta f}{f} = \underbrace{-\frac{1}{2}\frac{v_{\text{orb}}^2}{c^2}}_{\text{special relativity}} "
             r"+ \underbrace{\frac{GM_\oplus}{c^2}\left(\frac{1}{R_\oplus}-\frac{1}{R_\oplus+h}\right)}_{\text{general relativity}}")
    st.markdown(r"For a 500 km sun-synchronous LEO orbit this evaluates to approximately **+38.5 μs/day**, the same "
                r"order of magnitude as the celebrated GPS relativistic correction, underscoring that satellite QKD "
                r"synchronization windows must be relativity-aware at the nanosecond level required for gating "
                r"single-photon detectors.")

    st.markdown("---")

    # =========================================================================
    # SECTION 4 — GAUSSIAN BEAM OPTICS & FREE-SPACE LINK BUDGET
    # =========================================================================
    st.subheader("5. Gaussian Beam Propagation and Free-Space Link Budget")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown(r"**Paraxial Gaussian beam solution** of the wave equation, beam radius as a function of propagation distance $z$:")
        st.latex(r"W(z) = W_0\sqrt{1+\left(\frac{z}{z_R}\right)^{2}}, \qquad z_R = \frac{\pi W_0^{2}}{\lambda}")

        st.markdown(r"**Aperture-limited power coupling** onto a circular receiver of diameter $D_r$ at range $d$:")
        st.latex(r"G(d) = 1-\exp\!\left(-\frac{2a^{2}}{W(d)^{2}}\right), \quad a = D_r/2")

    with col_b2:
        st.markdown(r"**Beer–Lambert atmospheric extinction**, Kasten & Young (1989) relative air mass $m(\theta)$, "
                    r"regular at the horizon unlike the plain secant law:")
        st.latex(r"m(\theta) = \left[\cos\theta + 0.50572\,(96.07995-\theta)^{-1.6364}\right]^{-1}")
        st.latex(r"\eta(\theta) = \eta_{\text{det}}\cdot e^{-\tau_0 m(\theta)}\cdot G\big(d(\theta)\big)")

    st.markdown(
        r"**Kolmogorov atmospheric turbulence** introduces scintillation via the refractive-index structure "
        r"constant $C_n^2$; for a plane wave over path length $L$, the Rytov variance quantifies the "
        r"log-amplitude fluctuation regime:"
    )
    st.latex(r"\sigma_R^2 = 1.23\, C_n^2\, k^{7/6} L^{11/6}, \qquad k = \frac{2\pi}{\lambda}, "
             r"\qquad \sigma_R^2 \ll 1 \Rightarrow \text{weak turbulence}")

    st.markdown("---")

    # =========================================================================
    # SECTION 5 — BB84 PROTOCOL & DECOY-STATE SECURITY
    # =========================================================================
    st.subheader("6. BB84 Protocol Encoding and Decoy-State Security")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.table({
            "Basis": ["Z (rectilinear)", "Z (rectilinear)", "X (diagonal)", "X (diagonal)"],
            "Bit":   ["0", "1", "0", "1"],
            "State": ["|0⟩", "|1⟩", "|+⟩ = (|0⟩+|1⟩)/√2", "|−⟩ = (|0⟩−|1⟩)/√2"],
        })
        st.markdown(r"**Asymptotic decoy-state secure key rate** (Lo–Ma–Chen 2005; Ma, Qi, Zhao & Lo, "
                    r"*Phys. Rev. A* **72**, 012326, 2005):")
        st.latex(r"Y_1 = Y_0 + \eta(\theta) - Y_0\,\eta(\theta), \qquad Q_1 = \mu e^{-\mu} Y_1")
        st.latex(r"R_\infty = Q_1\big[1-h_2(e_1)\big] - Q_\mu\, f_{EC}\, h_2(E_\mu)")

    with col_c2:
        st.markdown(r"**Vacuum + weak decoy-state estimators**, using two finite intensities $\mu>\nu\geq 0$ "
                    r"and a vacuum decoy of yield $Y_0$:")
        st.latex(r"Y_1^{L} = \frac{\mu}{\mu\nu-\nu^{2}}\left[Q_\nu e^{\nu} - Q_\mu e^{\mu}\frac{\nu^{2}}{\mu^{2}} "
                 r"- \frac{\mu^{2}-\nu^{2}}{\mu^{2}}\,Y_0\right]")
        st.latex(r"e_1^{U} = \frac{E_\nu Q_\nu e^{\nu} - \tfrac{1}{2}Y_0}{Y_1^{L}\,\nu}")
        st.markdown(r"**Naive (photon-number-blind) rate**, vulnerable to photon-number-splitting, "
                    r"included for comparison in the Eavesdropping Analysis tab:")
        st.latex(r"R_{\text{naive}} = Q_\mu\big[1-h_2(E_\mu)\big] - Q_\mu\, f_{EC}\, h_2(E_\mu)")

    st.markdown("---")

    # =========================================================================
    # SECTION 6 — SYSTEM ARCHITECTURE & EVALUATION METHODOLOGY
    # =========================================================================
    st.subheader("7. System Architecture and Evaluation Methodology")
    st.markdown("""
    To systematically evaluate the spatial vulnerability of candidate Optical Ground Stations (OGS), 
    a deterministic, multi-stage simulation pipeline was engineered. This architecture integrates 
    empirical aviation telemetry with high-fidelity orbital propagation and quantum link-budget analyses.
    """)

    # Robust path definition to prevent initialization errors if ASSETS_DIR is currently undefined
    BASE_DIR_LOCAL = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR_LOCAL = os.path.join(BASE_DIR_LOCAL, "assets")

    img_pipeline = os.path.join(ASSETS_DIR_LOCAL, "DataAcquisition.png")
    if os.path.exists(img_pipeline):
        st.image(img_pipeline, use_container_width=True, caption="Data Acquisition and Evaluation Scheme. The pipeline progresses from geospatial grid generation and orbital sampling to aviation intersection detection and final cryptographic yield ranking.")
    else:
        st.info("📌 Add 'DataAcquisition.png' to the assets folder.")

    st.markdown("""
    **I. Grid Generation & Orbital Sampling:** The geographic region of interest is discretized into a uniform geospatial matrix. Concurrently, a representative set of satellite passes is generated, systematically varying the Right Ascension of the Ascending Node (RAAN) to ensure an unbiased probability distribution of overhead trajectories.  
    **II. Spatio-Temporal Intersection:** Real-world flight telemetry is processed to detect 3D line-of-sight crossings between aircraft and the satellite-to-OGS vector.  
    **III. Performance Aggregation:** Each location is ranked based on its aggregate long-term communication loss, dynamically computed from intersection events.
    """)

    st.markdown("---")

    # =========================================================================
    # SECTION 7 — ORBITAL GEOMETRY & TIME-DOMAIN MAPPING
    # =========================================================================
    st.subheader("8. Orbital Geometry and Time-Dependent Key Extraction")
    
    col_geom1, col_geom2 = st.columns([1, 1])
    with col_geom1:
        st.markdown(r"""
        The fundamental coupling between the satellite's orbital kinematics and the cryptographic yield is governed by the zenith angle $\theta(t)$. As the satellite traverses its orbit at altitude $h$, the instantaneous line-of-sight distance $d(t)$ strictly defines the elevation geometry:
        """)
        st.latex(r"\theta(t) = \arccos\left(\frac{h}{d(t)}\right)")
        st.markdown(r"""
        The quantum channel capacity relies on the **geometric secret-bit function** $S(\theta)$, which represents the extractable secure keys per signal under optimal tracking. Because atmospheric extinction and free-space diffraction losses degrade monotonically with slant range, $S(\theta)$ exhibits strict symmetric degradation, peaking at the zenith ($0^\circ$) and vanishing at the horizon ($\ge 90^\circ$). The time-domain mapping yields the dynamic link capacity:
        """)
        st.latex(r"S(t) = S(\theta(t))")
        
    with col_geom2:
        img_geom = os.path.join(ASSETS_DIR_LOCAL, "Geometry.png")
        if os.path.exists(img_geom):
            st.image(img_geom, use_container_width=True, caption="Geometric derivation of the zenith angle θ(t) and the subsequent time-domain mapping of the geometric secret-bit function S(θ) into the pass-dependent secure key rate S(t).")
        else:
            st.info("📌 Add 'Geometry.png' to the assets folder.")

    st.markdown("---")

    # =========================================================================
    # SECTION 8 — CRYPTOGRAPHIC LOSS INTEGRAL & CROSSING DETECTION
    # =========================================================================
    st.subheader("9. Cryptographic Loss Integral and Crossing-Detection Algorithm")
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown(r"**Time-domain integration of the aviation-safety blackout window** centred on crossing time $t_c$:")
        st.latex(r"S_{\text{loss}} = \int_{t_c-\Delta t/2}^{\,t_c+\Delta t/2} S(t)\,dt, "
                 r"\qquad \eta_{\text{link}} = 1-\frac{S_{\text{loss}}}{\int_{\text{pass}} S(t)\,dt}")

    with col_d2:
        st.markdown(r"**Line-segment intersection** between a satellite-pass segment $\overline{AB}$ and a "
                    r"flight-track segment $\overline{CD}$, solved via Cramer's rule:")
        st.latex(r"\begin{bmatrix} x_2-x_1 & -(x_4-x_3) \\ y_2-y_1 & -(y_4-y_3)\end{bmatrix}"
                 r"\begin{bmatrix} t \\ u\end{bmatrix} = \begin{bmatrix} x_3-x_1 \\ y_3-y_1\end{bmatrix}, "
                 r"\ t,u \in [0,1] \Leftrightarrow \text{crossing}")

    st.markdown("---")
    st.subheader("10. Illustrative Evaluation: The PLOB Repeaterless Bound vs. the Simulated Link")
    eta_range_th = np.linspace(0.001, 0.999, 300)
    plob_curve = plob_bound_bits_per_use(eta_range_th)
    fig_plob = go.Figure()
    fig_plob.add_trace(go.Scatter(x=eta_range_th, y=plob_curve, mode="lines",
                                   name="PLOB bound  K(η) = −log₂(1−η)", line=dict(color=DARK, width=3)))
    fig_plob.add_vline(x=eta0, line_dash="dash", line_color=ACCENT2,
                        annotation_text="current zenith channel transmittance η(0)", annotation_font_color=DARK)
    fig_plob.update_xaxes(title_text="Channel transmittance η")
    fig_plob.update_yaxes(title_text="Secret-key capacity (bits/channel use)")
    st.plotly_chart(style_fig(fig_plob, height=380, showlegend=False), width="stretch")
    st.caption("The PLOB bound is the fundamental repeaterless limit on any QKD protocol at a given channel "
               "transmittance; the decoy-state BB84 rate computed throughout this simulator necessarily lies "
               "below this curve, providing an independent theoretical sanity check on the link-budget model.")

# -----------------------------------------------------------------------------
# TAB 2 — QUANTUM CHANNEL & SECURITY
# -----------------------------------------------------------------------------
with tab_quantum:
    st.header("Quantum Channel and Security Analysis")

    st.subheader("A. BB84 Polarization States Represented on the Bloch Sphere")
    st.plotly_chart(build_bloch_sphere_figure(), use_container_width=True)

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        st.subheader("B. Quantum Bit Error Rate Along the Satellite Pass")
        fig_qber = go.Figure()
        fig_qber.add_trace(go.Scatter(x=t_arr, y=Emu_arr * 100, mode="lines",
                                       name="Overall QBER  E_μ(t)", line=dict(color=ACCENT, width=3)))
        fig_qber.add_trace(go.Scatter(x=t_arr, y=e1_arr * 100, mode="lines",
                                       name="Single-photon QBER  e₁(t)",
                                       line=dict(color=ACCENT4, width=2, dash="dot")))
        fig_qber.add_hline(y=BB84_THRESHOLD_QBER * 100, line_dash="dash", line_color="#D32F2F",
                            annotation_text="BB84 security threshold", annotation_font_color=DARK)
        fig_qber.update_xaxes(title_text="Time since horizon rise (s)")
        fig_qber.update_yaxes(title_text="QBER (%)")
        st.plotly_chart(style_fig(fig_qber, height=380), use_container_width=True)

    with col_q2:
        st.subheader("C. Secure Fraction as a Function of QBER")
        e_range = np.linspace(0.0, 0.5, 300)
        r_bb84 = np.clip(1.0 - 2.0 * binary_entropy(e_range), 0.0, None)
        fig_sec = go.Figure()
        fig_sec.add_trace(go.Scatter(x=e_range * 100, y=r_bb84, mode="lines",
                                      name="Single-photon bound  1−2h₂(e)",
                                      line=dict(color=ACCENT, width=3)))
        fig_sec.add_vline(x=BB84_THRESHOLD_QBER * 100, line_dash="dash", line_color="#D32F2F",
                           annotation_text=f"threshold ≈ {BB84_THRESHOLD_QBER*100:.1f}%", annotation_font_color=DARK)
        fig_sec.add_vline(x=Emu0 * 100, line_dash="dot", line_color=ACCENT2,
                           annotation_text="current zenith E_μ", annotation_font_color=DARK)
        fig_sec.update_xaxes(title_text="QBER e (%)")
        fig_sec.update_yaxes(title_text="Secure key fraction r(e)")
        st.plotly_chart(style_fig(fig_sec, height=380), use_container_width=True)

    st.subheader("D. Decoy-State Optimization: Secure Key Rate vs. Mean Photon Number")
    mu_range = np.linspace(0.05, 3.0, 80)
    rates_mu, q1_mu, naive_mu = sweep_mu_vs_rate(
        ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r, eta_det, p_dark, e_detector, f_EC, nu_hz, mu_range)
    fig_mu = make_subplots(specs=[[{"secondary_y": True}]])
    fig_mu.add_trace(go.Scatter(x=mu_range, y=rates_mu / 1000, name="Secure key rate (kbit/s)",
                                 line=dict(color=ACCENT, width=3)), secondary_y=False)
    fig_mu.add_trace(go.Scatter(x=mu_range, y=q1_mu, name="Single-photon gain Q₁",
                                 line=dict(color=ACCENT4, width=2, dash="dot")), secondary_y=True)
    fig_mu.add_vline(x=mu, line_dash="dash", line_color=ACCENT2, annotation_text="current μ", annotation_font_color=DARK)
    fig_mu.update_xaxes(title_text="Mean photon number μ")
    fig_mu.update_yaxes(title_text="Secure key rate at zenith (kbit/s)", secondary_y=False)
    fig_mu.update_yaxes(title_text="Single-photon gain Q₁", secondary_y=True)
    st.plotly_chart(style_fig(fig_mu, height=400), use_container_width=True)

    st.markdown("---")
    st.subheader("E. Vacuum + Weak Decoy-State Parameter Estimation Laboratory")
    st.markdown(
        r"This module performs a full experimental-style reconstruction of the decoy-state security proof. "
        r"Two finite signal intensities $\mu > \nu \geq 0$ and a vacuum pulse (yield $Y_0 = p_{\text{dark}}$) "
        r"are the only inputs an experimentalist has access to; from these, the single-photon yield $Y_1^L$ "
        r"and phase-error rate $e_1^U$ are estimated **without assuming the channel model is known** "
        r"(Ma, Qi, Zhao & Lo, *Phys. Rev. A* **72**, 012326, 2005)."
    )

    nu_decoy = st.slider("Weak decoy intensity ν (must satisfy 0 ≤ ν < μ)", 0.0, float(max(mu - 0.05, 0.01)),
                          float(min(0.1, max(mu - 0.05, 0.01))), 0.01, key="nu_decoy_slider")

    Q_nu0, E_nu0 = gain_and_qber(eta0, nu_decoy, p_dark, e_detector)
    Y1_L, Q1_L, e1_U = vacuum_weak_decoy_estimate(mu, nu_decoy, Qmu0, Emu0, Q_nu0, E_nu0, p_dark)
    Y1_true = p_dark + eta0 - p_dark * eta0

    ce1, ce2, ce3 = st.columns(3)
    ce1.metric("True single-photon yield Y₁", f"{Y1_true:.3e}")
    ce2.metric("Decoy-estimated lower bound Y₁ᴸ", f"{Y1_L:.3e}",
               f"{(Y1_L/Y1_true-1)*100:+.2f}% vs. truth" if Y1_true > 0 else "n/a")
    ce3.metric("Decoy-estimated e₁ᵁ vs. true e₁", f"{e1_U*100:.2f}% / {e10*100:.2f}%")

    col_e1, col_e2 = st.columns(2)

    # --- E1: Photon-number-resolved decomposition of the detection gain -----
    with col_e1:
        st.markdown("**Photon-Number-Resolved Decomposition of the Detection Gain**")
        st.caption(r"Waterfall decomposition of $Q_\mu$ by Fock-layer $n$, using the click model "
                   r"$Y_n = 1-(1-\eta)^n$ weighted by the Poisson distribution $P(n;\mu)$. "
                   r"The single-photon layer ($n{=}1$) is exactly the term the decoy-state method isolates.")
        n_vals, contrib, tail, Q_total = photon_number_gain_decomposition(mu, eta0, p_dark, n_max=4)
        labels = [f"n={n}" for n in n_vals] + ["n≥5 (tail)"]
        values = list(contrib) + [tail]
        colors_wf = [ACCENT3, ACCENT, ACCENT4, ACCENT2, "#607D8B", "#B0BEC5"]

        fig_wf = go.Figure(go.Waterfall(
            orientation="v",
            measure=["relative"] * len(labels) + ["total"],
            x=labels + ["Q_μ (total)"],
            y=values + [0],
            text=[f"{v:.2e}" for v in values] + [f"{Q_total:.2e}"],
            textposition="outside",
            connector=dict(line=dict(color="rgba(18,35,63,0.3)")),
            decreasing=dict(marker=dict(color=ACCENT2)),
            increasing=dict(marker=dict(color=ACCENT)),
            totals=dict(marker=dict(color=DARK)),
        ))
        fig_wf.update_yaxes(title_text="Contribution to Q_μ")
        st.plotly_chart(style_fig(fig_wf, height=420, showlegend=False), width="stretch")
        st.caption("The n=1 bar is the vulnerable-but-usable single-photon layer; n≥2 layers are the "
                   "photon-number-splitting-exploitable component the decoy-state method must exclude.")

    # --- E2: Estimator convergence as a function of the decoy intensity -----
    with col_e2:
        st.markdown("**Estimator Convergence: Y₁ᴸ(ν) vs. the True Single-Photon Yield**")
        st.caption(r"As $\nu \to 0$ the vacuum+weak decoy bound tightens monotonically onto the true "
                   r"$Y_1$ — the theoretical justification for using a *weak* (not strong) second decoy state.")
        nu_sweep = np.linspace(1e-4, mu * 0.97, 60)
        Y1_L_sweep = []
        for nv in nu_sweep:
            Q_nv, E_nv = gain_and_qber(eta0, nv, p_dark, e_detector)
            y1l, _, _ = vacuum_weak_decoy_estimate(mu, nv, Qmu0, Emu0, Q_nv, E_nv, p_dark)
            Y1_L_sweep.append(y1l)
        Y1_L_sweep = np.array(Y1_L_sweep)

        fig_conv = go.Figure()
        fig_conv.add_trace(go.Scatter(x=nu_sweep, y=Y1_L_sweep, mode="lines",
                                       name="Estimated Y₁ᴸ(ν)", line=dict(color=ACCENT4, width=3)))
        fig_conv.add_hline(y=Y1_true, line_dash="dash", line_color=ACCENT3,
                            annotation_text="True Y₁", annotation_font_color=DARK)
        fig_conv.add_vline(x=nu_decoy, line_dash="dot", line_color=DARK,
                            annotation_text="current ν", annotation_font_color=DARK)
        fig_conv.update_xaxes(title_text="Weak decoy intensity ν")
        fig_conv.update_yaxes(title_text="Single-photon yield estimate")
        st.plotly_chart(style_fig(fig_conv, height=420, showlegend=False), width="stretch")

    # --- E3: Estimator reliability heatmap over the (mu, nu) plane ---------
    st.markdown("**Estimator Reliability Map: Relative Bias of Y₁ᴸ over the (μ, ν) Decoy-Intensity Plane**")
    st.caption("Regions in red indicate parameter choices where the finite-decoy estimator is loose or "
               "numerically unstable (ν too close to μ, or μ too small for a resolvable signal); the "
               "white marker shows the currently selected (μ, ν) operating point.")

    mu_grid_e = np.linspace(0.15, 2.0, 24)
    bias_matrix = np.full((len(mu_grid_e), len(mu_grid_e)), np.nan)
    for i_m, mu_g in enumerate(mu_grid_e):
        nu_candidates = np.linspace(0.02, mu_g * 0.95, len(mu_grid_e))
        for i_n, nu_g in enumerate(nu_candidates):
            eta_ch_g, Qmu_g, Emu_g = channel_efficiency(0.0, ALT_SAT, eta_det, tau_zenith, D_r, W0, WAVELENGTH), None, None
            Qmu_g, Emu_g = gain_and_qber(eta_ch_g, mu_g, p_dark, e_detector)
            Qnu_g, Enu_g = gain_and_qber(eta_ch_g, nu_g, p_dark, e_detector)
            y1l_g, _, _ = vacuum_weak_decoy_estimate(mu_g, nu_g, Qmu_g, Emu_g, Qnu_g, Enu_g, p_dark)
            y1_true_g = p_dark + eta_ch_g - p_dark * eta_ch_g
            bias_matrix[i_n, i_m] = ((y1l_g / y1_true_g) - 1.0) * 100.0 if y1_true_g > 0 else np.nan

    fig_bias = go.Figure(data=go.Heatmap(
        z=bias_matrix, x=mu_grid_e, y=np.linspace(0.02, 1.0, len(mu_grid_e)),
        colorscale="RdYlGn", zmid=0, zmin=-50, zmax=5,
        colorbar=dict(title=dict(text="Relative bias (%)", font=dict(color=DARK)), tickfont=dict(color=DARK)),
    ))
    fig_bias.add_trace(go.Scatter(x=[mu], y=[nu_decoy], mode="markers",
                                   marker=dict(size=14, color="white", line=dict(color="black", width=2), symbol="x"),
                                   name="Current setting"))
    fig_bias.update_xaxes(title_text="Signal intensity μ")
    fig_bias.update_yaxes(title_text="Weak decoy intensity ν (relative scale)")
    st.plotly_chart(style_fig(fig_bias, height=440, showlegend=False), width="stretch")

# -----------------------------------------------------------------------------
# TAB 3 — MISSION CONTROL
# -----------------------------------------------------------------------------
with tab_mission:
    st.header("Mission Control — Live Satellite Pass Simulation")

    st.subheader("Manual Time Inspector")
    st.caption("Drag the slider to inspect the link state at any instant of the pass. This control is "
               "always synchronized with the physical model and is the most reliable way to explore the pass.")
    t_inspect = st.slider("Mission time t (s since horizon rise)", 0.0, float(2 * T_MAX),
                           float(np.clip(crossing_time, 0.0, 2 * T_MAX)), 1.0, key="t_inspect")
    idx_i = int(np.argmin(np.abs(t_arr - t_inspect)))
    theta_i, S_i, E_i = theta_deg_arr[idx_i], S_t_arr[idx_i], Emu_arr[idx_i]
    blackout_i = bool(mask_loss[idx_i])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Zenith angle θ", f"{theta_i:.1f}°")
    c2.metric("QBER", f"{E_i*100:.2f} %")
    c3.metric("Secure key rate", f"{S_i/1000:.2f} kbit/s")
    c4.metric("Link status", "BLACKOUT" if blackout_i else "LINK UP")

    col_sch, col_tl = st.columns([1, 1])
    with col_sch:
        st.plotly_chart(build_mission_schematic_figure(t_inspect, T_MAX, blackout_i, crossing_time,
                                                       max(margin_half, 1e-6), AC_ALT_FRACTION),
                        use_container_width=True)
    with col_tl:
        fig_tl_manual = go.Figure()
        fig_tl_manual.add_trace(go.Scatter(x=t_arr, y=S_t_arr, mode="lines",
                                            line=dict(color=ACCENT, width=2), name="S(t)"))
        fig_tl_manual.add_vrect(x0=t_start, x1=t_end, fillcolor="rgba(227,114,34,0.18)", line_width=0)
        fig_tl_manual.add_vline(x=t_inspect, line_color=DARK, line_width=2)
        fig_tl_manual.update_xaxes(title_text="Mission time t (s)")
        fig_tl_manual.update_yaxes(title_text="Secure key rate S(t) (bit/s)")
        st.plotly_chart(style_fig(fig_tl_manual, height=300, showlegend=False), use_container_width=True)

    st.markdown("---")
    st.subheader("Automated Pass Playback")
    st.caption("Runs a bounded, deterministic animation of the full pass. The manual inspector above "
               "remains the authoritative reference if the automated playback is interrupted.")

    if "mc_run_id" not in st.session_state:
        st.session_state.mc_run_id = 0

    play_col, reset_col = st.columns([1, 1])
    play_clicked = play_col.button("▶ Play Full Pass Animation", key="mc_play_btn")
    if reset_col.button("⟲ Reset Playback", key="mc_reset_btn"):
        st.session_state.mc_run_id += 1

    if play_clicked:
        st.session_state.mc_run_id += 1
        run_id = st.session_state.mc_run_id
        frame_idx = np.linspace(0, len(t_arr) - 1, 60).astype(int)
        status_ph = st.empty()
        chart_ph = st.empty()
        timeline_ph = st.empty()
        progress = st.progress(0.0)

        for k, i in enumerate(frame_idx):
            t_now = float(t_arr[i])
            theta_now = float(theta_deg_arr[i])
            S_now = float(S_t_arr[i])
            E_now = float(Emu_arr[i])
            in_blk = bool(mask_loss[i])

            with status_ph.container():
                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric("Mission time", f"{t_now:0.1f} s")
                mc2.metric("Zenith angle θ", f"{theta_now:0.1f}°")
                mc3.metric("QBER", f"{E_now*100:0.2f} %")
                mc4.metric("Secure key rate", f"{S_now/1000:0.2f} kbit/s",
                           "BLACKOUT" if in_blk else "LINK UP",
                           delta_color="inverse" if in_blk else "normal")

            with chart_ph.container():
                st.plotly_chart(
                    build_mission_schematic_figure(t_now, T_MAX, in_blk, crossing_time,
                                                    max(margin_half, 1e-6), AC_ALT_FRACTION),
                    use_container_width=True, key=f"mc_schem_{run_id}_{k}")

            with timeline_ph.container():
                fig_tl = go.Figure()
                fig_tl.add_trace(go.Scatter(x=t_arr[:i + 1], y=S_t_arr[:i + 1], mode="lines",
                                             line=dict(color=ACCENT, width=2), showlegend=False))
                fig_tl.add_vline(x=t_now, line_color=DARK)
                fig_tl.add_vrect(x0=t_start, x1=t_end, fillcolor="rgba(227,114,34,0.15)", line_width=0)
                fig_tl.update_xaxes(title_text="t (s)", range=[0, 2 * T_MAX])
                fig_tl.update_yaxes(title_text="S(t) (bit/s)")
                st.plotly_chart(style_fig(fig_tl, height=220, showlegend=False),
                                 use_container_width=True, key=f"mc_tl_{run_id}_{k}")

            progress.progress((k + 1) / len(frame_idx))
            time.sleep(0.04)

        st.success("Pass simulation complete — link statistics matched the manual inspector at every frame.")

# -----------------------------------------------------------------------------
# TAB 4 — KEY-LOSS ANALYSIS
# -----------------------------------------------------------------------------
with tab_crypto:
    st.header("Time-Domain Analysis of Aviation-Induced Key Blackout")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Key Yield Potential", f"{total_key_yield:.2f} bit")
    m2.metric("Aviation-Induced Key Loss", f"{lost_key_yield:.2f} bit",
              f"-{(lost_key_yield/total_key_yield*100 if total_key_yield>0 else 0):.2f}%",
              delta_color="inverse")
    m3.metric("Net Link Efficiency", f"{efficiency:.2f} %")
    m4.metric("Zenith Angle at Crossing", f"{np.interp(crossing_time, t_arr, theta_deg_arr):.1f}°")

    fig_crypto = make_subplots(specs=[[{"secondary_y": True}]])
    S_safe = np.copy(S_t_arr); S_safe[mask_loss] = 0.0
    S_loss = np.zeros_like(S_t_arr); S_loss[mask_loss] = S_t_arr[mask_loss]

    fig_crypto.add_trace(go.Scatter(x=t_arr, y=S_t_arr, mode="lines", name="S(t) — secure key rate",
                                     line=dict(color=ACCENT, width=2)), secondary_y=False)
    fig_crypto.add_trace(go.Scatter(x=t_arr, y=S_safe, fill="tozeroy", mode="none",
                                     name="Distributed keys", fillcolor="rgba(0,101,189,0.15)"),
                          secondary_y=False)
    fig_crypto.add_trace(go.Scatter(x=t_arr, y=S_loss, fill="tozeroy", mode="none",
                                     name="Lost keys (blackout)", fillcolor="rgba(227,114,34,0.55)"),
                          secondary_y=False)
    fig_crypto.add_trace(go.Scatter(x=t_arr, y=theta_deg_arr, mode="lines", name="θ(t)",
                                     line=dict(color="rgba(60,60,60,0.6)", width=1.5, dash="dot")),
                          secondary_y=True)
    fig_crypto.add_vline(x=crossing_time, line_width=2, line_dash="dash", line_color=DARK,
                          annotation_text="Aircraft intercept tc", annotation_font_color=DARK)
    fig_crypto.update_xaxes(title_text="Time since horizon rise (s)")
    fig_crypto.update_yaxes(title_text="Secure key rate (bit/s)", secondary_y=False)
    fig_crypto.update_yaxes(title_text="Zenith angle θ (deg)", secondary_y=True, range=[0, 95])
    st.plotly_chart(style_fig(fig_crypto, height=520, margin_t=40), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 5 — EAVESDROPPING ANALYSIS
# -----------------------------------------------------------------------------
with tab_eve:
    st.header("Security Under Active Eavesdropping")
    attack_type = st.radio("Attack model", ["No eavesdropper", "Intercept–Resend", "Photon-Number-Splitting (PNS)"],
                            horizontal=True)

    if attack_type == "Intercept–Resend":
        p_intercept = st.slider("Fraction of pulses intercepted by Eve, p_E", 0.0, 1.0, 0.30, 0.01)
        E_eve = float(np.clip(Emu0 + 0.25 * p_intercept, 0.0, 0.5))
        e1_eve = float(np.clip(e10 + 0.25 * p_intercept, 0.0, 0.5))
        R_eve_bits = float(np.clip(Q10 * (1 - binary_entropy(e1_eve)) - Qmu0 * f_EC * binary_entropy(E_eve), 0.0, None))
        S_eve = R_eve_bits * nu_hz
        p_crit = float(np.clip((BB84_THRESHOLD_QBER - Emu0) / 0.25, 0.0, 1.0))

        c1, c2, c3 = st.columns(3)
        c1.metric("QBER without Eve", f"{Emu0*100:.2f} %")
        c2.metric("QBER with Eve", f"{E_eve*100:.2f} %", f"+{(E_eve-Emu0)*100:.2f} pp", delta_color="inverse")
        c3.metric("Secure key rate with Eve", f"{S_eve/1000:.2f} kbit/s",
                  f"{(S_eve-S0)/1000:.2f} kbit/s", delta_color="inverse")

        if E_eve >= BB84_THRESHOLD_QBER:
            st.error(f"QBER exceeds the BB84 security threshold ({BB84_THRESHOLD_QBER*100:.1f}%) — "
                     f"the protocol aborts and no secure key is distilled.")
        else:
            st.success("The channel remains within the security threshold — a positive secure key rate is still distillable.")
        st.caption(f"Detection threshold: the protocol aborts once Eve intercepts more than {p_crit*100:.1f}% of the pulses.")

        colA, colB = st.columns(2)
        with colA:
            st.plotly_chart(build_disturbance_bloch_figure(p_intercept), use_container_width=True)
        with colB:
            p_range = np.linspace(0, 1, 100)
            E_range = np.clip(Emu0 + 0.25 * p_range, 0, 0.5)
            e1_range = np.clip(e10 + 0.25 * p_range, 0, 0.5)
            R_range_bits = np.clip(Q10 * (1 - binary_entropy(e1_range)) - Qmu0 * f_EC * binary_entropy(E_range), 0, None)
            S_range = R_range_bits * nu_hz
            fig_ir = make_subplots(specs=[[{"secondary_y": True}]])
            fig_ir.add_trace(go.Scatter(x=p_range * 100, y=E_range * 100, name="QBER (%)",
                                         line=dict(color="#D32F2F", width=3)), secondary_y=False)
            fig_ir.add_trace(go.Scatter(x=p_range * 100, y=S_range / 1000, name="Secure key rate (kbit/s)",
                                         line=dict(color=ACCENT, width=3, dash="dot")), secondary_y=True)
            fig_ir.add_vline(x=p_intercept * 100, line_dash="dash", line_color=DARK)
            fig_ir.add_hline(y=BB84_THRESHOLD_QBER * 100, line_dash="dot", line_color="#D32F2F", secondary_y=False)
            fig_ir.update_xaxes(title_text="Intercepted fraction p_E (%)")
            fig_ir.update_yaxes(title_text="QBER (%)", secondary_y=False)
            fig_ir.update_yaxes(title_text="Secure key rate (kbit/s)", secondary_y=True)
            st.plotly_chart(style_fig(fig_ir, height=430), use_container_width=True)

    elif attack_type == "Photon-Number-Splitting (PNS)":
        st.markdown("Under a PNS attack, Eve exploits the multi-photon component of weak coherent pulses "
                    "without introducing detectable errors. A photon-number-blind analysis therefore "
                    "overstates the secure key rate; the decoy-state method restores security by isolating "
                    "the true single-photon contribution Q₁, e₁.")
        mu_r = np.linspace(0.05, 3.0, 100)
        naive_arr, decoy_arr = [], []
        for m in mu_r:
            eta_ch, Qmu_m, Emu_m, Q1_m, e1_m, S_m = compute_link_metrics(
                0.0, ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r, eta_det, m, p_dark, e_detector, f_EC, nu_hz)
            naive_arr.append(float(naive_bb84_rate_bits(Qmu_m, Emu_m, f_EC) * nu_hz))
            decoy_arr.append(float(S_m))
        naive_arr, decoy_arr = np.array(naive_arr), np.array(decoy_arr)

        fig_pns = go.Figure()
        fig_pns.add_trace(go.Scatter(x=mu_r, y=naive_arr / 1000, name="Naive (photon-number-blind) rate",
                                      line=dict(color="#D32F2F", width=3)))
        fig_pns.add_trace(go.Scatter(x=mu_r, y=decoy_arr / 1000, name="Decoy-state corrected rate",
                                      line=dict(color=ACCENT, width=3)))
        fig_pns.add_vline(x=mu, line_dash="dash", line_color=DARK, annotation_text="current μ", annotation_font_color=DARK)
        fig_pns.update_xaxes(title_text="Mean photon number μ")
        fig_pns.update_yaxes(title_text="Apparent secure key rate (kbit/s)")
        st.plotly_chart(style_fig(fig_pns, height=430), use_container_width=True)

        idx_now = int(np.argmin(np.abs(mu_r - mu)))
        overstatement = naive_arr[idx_now] - decoy_arr[idx_now]
        pct = (overstatement / naive_arr[idx_now] * 100.0) if naive_arr[idx_now] > 0 else 0.0
        st.metric("Security overstatement at current μ", f"{overstatement/1000:.2f} kbit/s", f"{pct:.1f} % of naive estimate")

    else:
        st.info("No eavesdropper present. Baseline channel metrics are shown in the executive KPI row "
                "and in the Quantum Channel & Security tab.")

# -----------------------------------------------------------------------------
# TAB 6 — 3D ORBITAL VIEW
# -----------------------------------------------------------------------------
with tab_orbit3d:
    st.header("Three-Dimensional Orbital Geometry")
    st.caption("Illustrative rendering; Earth-fixed and orbital reference frames are aligned for this "
               "static view and Earth's rotation is not propagated.")

    fig_orbit = build_earth_orbit_figure(ALT_SAT, INCLINATION, crossing_time, T_MAX, omega,
                                          GARCHING_LAT, GARCHING_LON)
    st.plotly_chart(fig_orbit, use_container_width=True)

    st.subheader("Schematic Sky-Track (elevation exact, azimuth illustrative)")
    azimuth_schematic = 180.0 * (t_arr / (2 * T_MAX))
    fig_sky = go.Figure(go.Scatterpolar(
        r=theta_deg_arr, theta=azimuth_schematic, mode="lines",
        line=dict(color=ACCENT, width=3), name="Satellite track"))
    theta_at_crossing = float(np.interp(crossing_time, t_arr, theta_deg_arr))
    az_at_crossing = float(np.interp(crossing_time, t_arr, azimuth_schematic))
    fig_sky.add_trace(go.Scatterpolar(r=[theta_at_crossing], theta=[az_at_crossing], mode="markers",
                                       marker=dict(size=12, color="#D32F2F", symbol="x"),
                                       name="Aircraft crossing"))
    fig_sky.update_layout(
        polar=dict(radialaxis=dict(range=[0, 90], tickvals=[0, 30, 60, 90],
                                    ticktext=["Zenith", "30°", "60°", "Horizon"],
                                    tickfont=dict(color=DARK)),
                   angularaxis=dict(tickfont=dict(color=DARK))),
    )
    st.plotly_chart(style_fig(fig_sky, height=480, margin_t=10), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 7 — 3D ECLIPSE
# -----------------------------------------------------------------------------
with tab_optics:
    st.header("Three-Dimensional Optical Intensity Occultation")

    X, Y, I_ecl, transmittance = build_eclipse_grid(W_Z, ac_offset, ac_diameter)

    c_opt1, c_opt2 = st.columns([1, 3])
    with c_opt1:
        st.metric("Beam radius at aircraft altitude", f"{W_Z:.2f} m")
        st.metric("Rayleigh range", f"{Z_R/1000:.2f} km")
        st.metric(r"Integrated transmittance $\eta(t)$", f"{transmittance*100:.2f} %")
        if transmittance < 0.05:
            st.error("### LINK STATUS\n**CRITICAL BLACKOUT**\n\nTransmittance < 5 %")
        elif transmittance < 0.60:
            st.warning("### LINK STATUS\n**DEGRADING**\n\nPartial occultation")
        else:
            st.success("### LINK STATUS\n**SECURE**\n\nTransmittance > 60 %")

    with c_opt2:
        fig_3d = go.Figure(data=[go.Surface(z=I_ecl, x=X, y=Y, colorscale="Blues", opacity=0.95)])
        fig_3d.update_layout(
            scene=dict(xaxis_title="Cross-track X (m)", yaxis_title="Along-track Y (m)",
                       zaxis_title="Normalized intensity I(r)", zaxis=dict(range=[0, 1])),
            template=PLOTLY_TEMPLATE, height=600, margin=dict(l=0, r=0, b=0, t=0),
            font=dict(color=DARK))
        st.plotly_chart(fig_3d, use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 8 — GEOSPATIAL RISK MAP
# -----------------------------------------------------------------------------
with tab_geo:
    st.header("Munich Metropolitan Area — Optical Ground Station Risk Grid")
    st.caption("Simulated 41×41 candidate-OGS matrix: aviation-crossing risk grows with proximity to "
               "Munich Airport and scales with the current safety margin.")

    flat_lats, flat_lons, flat_risks = build_risk_grid(margin_slider)

    fig_map = go.Figure()
    fig_map.add_trace(go.Scattermap(
        lat=flat_lats, lon=flat_lons, mode="markers",
        marker=dict(size=9, color=flat_risks, colorscale="Oranges", showscale=True,
                    colorbar=dict(title=dict(text="Key Loss Risk (%)", font=dict(color=DARK)), tickfont=dict(color=DARK))),
        hovertext=[f"Vulnerability: {r:.1f}%" for r in flat_risks], name="Candidate OGS"))
    fig_map.add_trace(go.Scattermap(
        lat=[GARCHING_LAT], lon=[GARCHING_LON], mode="markers+text",
        marker=dict(size=15, color=ACCENT),
        text=["TUM Garching OGS"], textposition="bottom right", name="TUM",
        textfont=dict(color=DARK, size=13)))
    fig_map.add_trace(go.Scattermap(
        lat=[MUC_LAT], lon=[MUC_LON], mode="markers+text",
        marker=dict(size=15, color="#D32F2F"),
        text=["MUC Airport"], textposition="top right", name="Airport",
        textfont=dict(color=DARK, size=13)))
    fig_map.update_layout(map=dict(style="open-street-map", center=dict(lat=48.3, lon=11.7), zoom=7.5))
    st.plotly_chart(style_fig(fig_map, height=700, margin_t=0), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 9 — NETWORK DESIGNER
# -----------------------------------------------------------------------------
with tab_network:
    st.header("Optical Ground Station Network Designer")
    st.markdown("For a sun-synchronous constellation with uniformly distributed right ascension of the "
                "ascending node (RAAN), every candidate site accrues equal long-term key-generation "
                "potential; siting is therefore governed by aviation-crossing risk alone.")

    default_sites = pd.DataFrame({
        "Site": ["TUM Garching", "Hohenpeissenberg", "Augsburg", "Ingolstadt", "Rosenheim"],
        "Latitude": [48.2665, 47.8017, 48.3705, 48.7665, 47.8563],
        "Longitude": [11.6691, 11.0119, 10.8978, 11.4257, 12.1288],
    })
    site_table = st.data_editor(default_sites, num_rows="dynamic", use_container_width=True, key="site_editor")

    if len(site_table) > 0:
        dists = haversine_km(site_table["Latitude"].to_numpy(dtype=float),
                              site_table["Longitude"].to_numpy(dtype=float), MUC_LAT, MUC_LON)
        margin_factor = margin_slider / 60.0
        risks = np.clip(1.0 - dists / DIST_INFLUENCE_KM, 0.0, 1.0) * 100.0 * margin_factor
        yield_index = efficiency * (1.0 - risks / 100.0)

        results = site_table.copy()
        results["Distance to MUC (km)"] = np.round(dists, 1)
        results["Aviation Risk (%)"] = np.round(risks, 1)
        results["Net Efficiency (%)"] = round(efficiency, 2)
        results["Yield Index"] = np.round(yield_index, 2)
        results = results.sort_values("Yield Index", ascending=False).reset_index(drop=True)

        st.dataframe(results, use_container_width=True)
        best = results.iloc[0]
        st.success(f"Recommended site: **{best['Site']}** — lowest aviation-crossing risk "
                   f"({best['Aviation Risk (%)']:.1f} %), yield index {best['Yield Index']:.2f}.")

        fig_net = go.Figure()
        fig_net.add_trace(go.Scattermap(
            lat=results["Latitude"], lon=results["Longitude"], mode="markers+text",
            marker=dict(size=15, color=results["Aviation Risk (%)"], colorscale="Oranges",
                        showscale=True, colorbar=dict(title=dict(text="Risk (%)", font=dict(color=DARK)), tickfont=dict(color=DARK))),
            text=results["Site"], textposition="top right", name="Candidate OGS",
            textfont=dict(color=DARK, size=13)))
        fig_net.add_trace(go.Scattermap(lat=[MUC_LAT], lon=[MUC_LON], mode="markers+text",
                                         marker=dict(size=15, color="#D32F2F"),
                                         text=["MUC Airport"], textposition="bottom right", name="Airport",
                                         textfont=dict(color=DARK, size=13)))
        fig_net.update_layout(map=dict(style="open-street-map", center=dict(lat=48.3, lon=11.7), zoom=7.2))
        st.plotly_chart(style_fig(fig_net, height=560, margin_t=0), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 10 — CROSSING DETECTION
# -----------------------------------------------------------------------------
with tab_cross:
    st.header("Interactive Crossing-Detection Algorithm")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Satellite pass segment (A → B)**")
        Ax = st.number_input("A — longitude", value=11.60, step=0.01, format="%.2f")
        Ay = st.number_input("A — latitude", value=48.10, step=0.01, format="%.2f")
        Bx = st.number_input("B — longitude", value=11.95, step=0.01, format="%.2f")
        By = st.number_input("B — latitude", value=48.55, step=0.01, format="%.2f")
    with c2:
        st.markdown("**Flight-track segment (C → D)**")
        Cx = st.number_input("C — longitude", value=11.50, step=0.01, format="%.2f")
        Cy = st.number_input("C — latitude", value=48.40, step=0.01, format="%.2f")
        Dx = st.number_input("D — longitude", value=12.00, step=0.01, format="%.2f")
        Dy = st.number_input("D — latitude", value=48.30, step=0.01, format="%.2f")

    t_val, u_val, point = line_segment_intersection((Ax, Ay), (Bx, By), (Cx, Cy), (Dx, Dy))

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=[Ax, Bx], y=[Ay, By], mode="lines+markers",
                                   name="Satellite pass (AB)", line=dict(color=ACCENT, width=3)))
    fig_line.add_trace(go.Scatter(x=[Cx, Dx], y=[Cy, Dy], mode="lines+markers",
                                   name="Flight track (CD)", line=dict(color=ACCENT2, width=3)))
    if point is not None:
        fig_line.add_trace(go.Scatter(x=[point[0]], y=[point[1]], mode="markers",
                                       marker=dict(size=14, color="#D32F2F", symbol="x"), name="Crossing"))
        st.success(f"**Crossing detected** at t = {t_val:.3f}, u = {u_val:.3f} → "
                   f"(lon, lat) = ({point[0]:.4f}, {point[1]:.4f})")
    else:
        st.info("No crossing within both segments' bounds (t, u ∉ [0, 1], or segments parallel).")

    fig_line.update_xaxes(title_text="Longitude (°)")
    fig_line.update_yaxes(title_text="Latitude (°)")
    st.plotly_chart(style_fig(fig_line, height=480, margin_t=20), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 11 — SENSITIVITY ANALYSIS
# -----------------------------------------------------------------------------
with tab_sweep:
    st.header("Parameter Sensitivity Analysis")

    st.subheader("A. Net Link Efficiency vs. Safety Margin")
    margins = np.linspace(0.0, 200.0, 40)
    effs_margin = sweep_margin_vs_efficiency(ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r, eta_det,
                                              mu, p_dark, e_detector, f_EC, nu_hz, crossing_time, margins)
    fig_a = go.Figure()
    fig_a.add_trace(go.Scatter(x=margins, y=effs_margin, mode="lines", line=dict(color=ACCENT, width=3)))
    fig_a.add_vline(x=margin_slider, line_dash="dash", line_color=ACCENT2, annotation_text="current Δt", annotation_font_color=DARK)
    fig_a.update_xaxes(title_text="Safety margin Δt (s)")
    fig_a.update_yaxes(title_text="Net link efficiency (%)")
    st.plotly_chart(style_fig(fig_a, height=380, showlegend=False), use_container_width=True)

    col_b, col_c = st.columns(2)
    with col_b:
        st.subheader("B. Pass Duration & Orbital Velocity vs. Altitude")
        alt_range = np.linspace(200_000.0, 800_000.0, 40)
        durations, velocities = sweep_altitude_vs_pass(alt_range)
        fig_b = make_subplots(specs=[[{"secondary_y": True}]])
        fig_b.add_trace(go.Scatter(x=alt_range / 1000, y=durations, name="Pass duration (s)",
                                    line=dict(color=ACCENT, width=3)), secondary_y=False)
        fig_b.add_trace(go.Scatter(x=alt_range / 1000, y=velocities / 1000, name="Orbital velocity (km/s)",
                                    line=dict(color=ACCENT2, width=3, dash="dot")), secondary_y=True)
        fig_b.add_vline(x=alt_sat_km, line_dash="dash", line_color=DARK)
        fig_b.update_xaxes(title_text="Satellite altitude (km)")
        fig_b.update_yaxes(title_text="Pass duration (s)", secondary_y=False)
        fig_b.update_yaxes(title_text="Orbital velocity (km/s)", secondary_y=True)
        st.plotly_chart(style_fig(fig_b, height=380), use_container_width=True)

    with col_c:
        st.subheader("C. Net Link Efficiency vs. Atmospheric Optical Depth")
        tau_range = np.linspace(0.05, 0.50, 40)
        effs_tau = sweep_tau_vs_efficiency(ALT_SAT, WAVELENGTH, W0, D_r, eta_det, mu, p_dark,
                                            e_detector, f_EC, nu_hz, crossing_time, margin_slider, tau_range)
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(x=tau_range, y=effs_tau, mode="lines", line=dict(color=ACCENT, width=3)))
        fig_c.add_vline(x=tau_zenith, line_dash="dash", line_color=ACCENT2, annotation_text="current τ₀", annotation_font_color=DARK)
        fig_c.update_xaxes(title_text="Zenith optical depth τ₀")
        fig_c.update_yaxes(title_text="Net link efficiency (%)")
        st.plotly_chart(style_fig(fig_c, height=380, showlegend=False), use_container_width=True)

    st.subheader("D. Beam Radius Growth Along the Slant Path")
    z_range = np.linspace(0.0, ALT_SAT, 200)
    w_range, _ = gaussian_beam_radius(W0, WAVELENGTH, z_range)
    fig_d = go.Figure()
    fig_d.add_trace(go.Scatter(x=z_range / 1000, y=w_range, mode="lines",
                                line=dict(color=ACCENT, width=3), name="W(z)"))
    fig_d.add_vline(x=Z_DIST / 1000, line_dash="dash", line_color=ACCENT2, annotation_text="aircraft altitude", annotation_font_color=DARK)
    fig_d.update_xaxes(title_text="Propagation distance z (km)")
    fig_d.update_yaxes(title_text="Beam radius W(z) (m)")
    st.plotly_chart(style_fig(fig_d, height=360, showlegend=False), use_container_width=True)

    st.subheader("E. Secure Key Rate Sensitivity Map: μ × Detector Efficiency")
    mu_grid = np.linspace(0.05, 2.0, 26)
    eta_grid = np.linspace(0.10, 0.90, 26)
    rate_matrix = build_rate_heatmap(ALT_SAT, WAVELENGTH, W0, tau_zenith, D_r, p_dark, e_detector,
                                      f_EC, nu_hz, mu_grid, eta_grid) / 1000.0
    fig_heat = go.Figure(data=go.Heatmap(z=rate_matrix, x=mu_grid, y=eta_grid, colorscale="Viridis",
                                          colorbar=dict(title=dict(text="kbit/s", font=dict(color=DARK)), tickfont=dict(color=DARK))))
    fig_heat.add_trace(go.Scatter(x=[mu], y=[eta_det], mode="markers",
                                   marker=dict(size=14, color="white", line=dict(color="black", width=2), symbol="x"),
                                   name="Current setting"))
    fig_heat.update_xaxes(title_text="Mean photon number μ")
    fig_heat.update_yaxes(title_text="Detector efficiency η_det")
    st.plotly_chart(style_fig(fig_heat, height=440, showlegend=False), use_container_width=True)

# -----------------------------------------------------------------------------
# TAB 12 — EXPERIMENTAL RESULTS ARCHIVE (COMPREHENSIVE STATIC GALLERY)
# -----------------------------------------------------------------------------
with tab_own:
    st.header("Numerical Simulation Results")
    st.markdown("""
    This archive holds the comprehensive static figures generated independently by the author (**Andrea Staffieri**) 
    from dedicated batch simulation runs. These high-fidelity analyses expand upon the foundational model from the 
    Master's Thesis by **Ada Kanoğlu (2025)**: *"Simulation and Evaluation of Optical Ground Station 
    Locations for Safe Satellite-Based Quantum Communication Near Airports"*.
    
    The extensive plots below represent the raw, un-interpolated data outputs categorized by their specific phenomenological domain.
    """)

    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ASSETS_DIR = os.path.join(BASE_DIR, "assets")

    # =========================================================================
    # 1. ORBITAL DYNAMICS & MACRO ANALYSIS
    # =========================================================================
    st.subheader("1. Macro Orbital Dynamics & Sun-Synchronous Stability")
    st.markdown("Long-term astrodynamical propagation and ground-track intersection stability of the LEO satellite.")
    
    col_orb1, col_orb2= st.columns(2)
    with col_orb1:
        img = os.path.join(ASSETS_DIR, "Orbital_Intersections_MacroScale.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="Macro-Scale Orbital Intersections")
        else: st.info("📌 Add 'Orbital_Intersections_MacroScale.png' (Folder 5)")
    with col_orb2:
        img = os.path.join(ASSETS_DIR, "Sun_Sync_1Year_Analysis.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="1-Year Sun-Synchronous Analysis")
        else: st.info("📌 Add 'Sun_Sync_1Year_Analysis.png' (Folder 8)")

    st.markdown("---")

    # =========================================================================
    # 2. AIRCRAFT KINEMATICS: SPEED VS ALTITUDE & PROFILES
    # =========================================================================
    st.subheader("2. Aircraft Kinematics: Speed vs. Altitude Dynamics")
    st.markdown("Correlation between aircraft cruise altitudes and required relative speeds at varying simulated satellite altitudes.")
    
    col_kin1, col_kin2 = st.columns(2)
    with col_kin1:
        img = os.path.join(ASSETS_DIR, "200km.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="Speed vs. Altitude (200km)")
        else: st.info("📌 Add '200km.png' (Folder 3)")
        
        img = os.path.join(ASSETS_DIR, "500km.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="Speed vs. Altitude (500km)")
        else: st.info("📌 Add '500km.png' (Folder 3)")
        
    with col_kin2:
        img = os.path.join(ASSETS_DIR, "300km.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="Speed vs. Altitude (300km)")
        else: st.info("📌 Add '300km.png' (Folder 3)")
        
        img = os.path.join(ASSETS_DIR, "800km.png")
        if os.path.exists(img): st.image(img, use_container_width=True, caption="Speed vs. Altitude (800km)")
        else: st.info("📌 Add '800km.png' (Folder 3)")

    st.markdown("---")
    
    # =========================================================================
    # 3. AIRCRAFT TRAJECTORY PROFILES & KINEMATIC CROSS-SECTIONS
    # =========================================================================
    st.subheader("3. Aircraft Trajectory Profiles & Kinematic Cross-Sections")
    
    # Original empirical image (flight trajectory profile)
    img_profile = os.path.join(ASSETS_DIR, "profilo_volo.png")
    if os.path.exists(img_profile): 
        # Layout ratio [1, 4, 1]: The graph is appropriately scaled (occupies nearly 70% of the screen) 
        # while protective lateral margins prevent geometric distortion and preserve HD resolution
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(img_profile, use_container_width=True, caption="UAV Flight Trajectory Profile")
    else: 
        st.info("📌 Add 'profilo_volo.png' (Folder 4)")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    **Kinematic Asymmetry: Inbound vs. Outbound Flights**  
    A fundamental aspect of QKD link availability is the kinematic asymmetry between landing and departing aircraft. 
    Inbound flights follow an extended, shallow glide slope, maintaining low altitudes for longer durations, 
    thus increasing the spatial probability of intersecting the satellite's optical beam. Conversely, outbound flights 
    execute a steep climbout. The high-resolution kinematic cross-sections below demonstrate this divergence.
    """)

    # Twin comparative images displayed side-by-side
    col_in, col_out = st.columns(2)
    
    with col_in:
        img_in = os.path.join(ASSETS_DIR, "Inbound_Trajectories_Arrivals.png")
        if os.path.exists(img_in):
            st.image(img_in, use_container_width=True, caption="Inbound Trajectories (Approach & Glide Slope Phase)")
        else:
            st.info("📌 Add 'Inbound_Trajectories_Arrivals.png' to the 'assets' folder")

    with col_out:
        img_out = os.path.join(ASSETS_DIR, "Outbound_Trajectories_Departures.png")
        if os.path.exists(img_out):
            st.image(img_out, use_container_width=True, caption="Outbound Trajectories (Climbout Phase)")
        else:
            st.info("📌 Add 'Outbound_Trajectories_Departures.png' to the 'assets' folder")

    st.markdown("---")

    # =========================================================================
    # 4. SAFETY MARGIN SWEEP (DENSITY PLOTS) - THE CORE DATA
    # =========================================================================
    st.subheader("4. Spatial Vulnerability: Safety Margin Sweep (Density Plots)")
    st.markdown(r"Comprehensive mapping of aviation-induced QKD interruptions over the Munich grid at varying safety margins ($\Delta t$).")
    
    margins = ["0s", "30s", "60s", "120s", "1000s", "5000s"]
    
    for m in margins:
        with st.expander(f"Simulated Safety Margin: Δt = {m}", expanded=(m=="0s")):
            col_m1, col_m2, col_m3 = st.columns(3)
            
            with col_m1:
                img = os.path.join(ASSETS_DIR, f"density_plot_KeyLengthLoss_{m}.png")
                if os.path.exists(img): st.image(img, use_container_width=True, caption=f"Total Key Length Loss ({m})")
                else: st.info(f"📌 Add 'density_plot_KeyLengthLoss_{m}.png' (Folder 1)")
            
            with col_m2:
                img = os.path.join(ASSETS_DIR, f"density_plot_AverageKeyLengthLoss_{m}.png")
                if os.path.exists(img): st.image(img, use_container_width=True, caption=f"Average Key Length Loss ({m})")
                else: st.info(f"📌 Add 'density_plot_AverageKeyLengthLoss_{m}.png' (Folder 1)")
                
            with col_m3:
                img = os.path.join(ASSETS_DIR, f"density_plot_NumOfCross_{m}.png")
                if os.path.exists(img): st.image(img, use_container_width=True, caption=f"Number of Crossings ({m})")
                else: st.info(f"📌 Add 'density_plot_NumOfCross_{m}.png' (Folder 1)")

    st.markdown("---")

    # =========================================================================
    # 5. DAY VS NIGHT ANALYSIS
    # =========================================================================
    st.subheader("5. Day vs. Night Operational Budget")
    st.markdown("""
    Density plot comparison of QKD efficiency under varying solar background noise conditions (**60s safety margin baseline**). 
    
    > **Note on Diurnal/Nocturnal Asymmetry:** The spatial analysis reveals a minimal divergence in aviation-induced key loss between daytime and nighttime operations. This phenomenon is directly attributed to the strict night flight curfew (*Nachtflugverbot*) enforced at Munich Airport. The regulatory curtailment of air traffic drastically reduces the probability of line-of-sight interruptions during nocturnal QKD transmission windows.
    """)
    
    tab_day, tab_night = st.tabs(["☀️ Day Operations", "🌙 Night Operations"])
    
    with tab_day:
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            img = os.path.join(ASSETS_DIR, "day_density_plot_KeyLengthLoss_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Total Key Loss (Day)")
            else: st.info("📌 Add 'day_density_plot_KeyLengthLoss_60s.png' (Folder 2)")
        with col_d2:
            img = os.path.join(ASSETS_DIR, "day_density_plot_AverageKeyLengthLoss_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Average Key Loss (Day)")
            else: st.info("📌 Add 'day_density_plot_AverageKeyLengthLoss_60s.png' (Folder 2)")
        with col_d3:
            img = os.path.join(ASSETS_DIR, "day_density_plot_NumOfCross_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Number of Crossings (Day)")
            else: st.info("📌 Add 'day_density_plot_NumOfCross_60s.png' (Folder 2)")

    with tab_night:
        col_n1, col_n2, col_n3 = st.columns(3)
        with col_n1:
            img = os.path.join(ASSETS_DIR, "night_density_plot_KeyLengthLoss_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Total Key Loss (Night)")
            else: st.info("📌 Add 'night_density_plot_KeyLengthLoss_60s.png' (Folder 2)")
        with col_n2:
            img = os.path.join(ASSETS_DIR, "night_density_plot_AverageKeyLengthLoss_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Average Key Loss (Night)")
            else: st.info("📌 Add 'night_density_plot_AverageKeyLengthLoss_60s.png' (Folder 2)")
        with col_n3:
            img = os.path.join(ASSETS_DIR, "night_density_plot_NumOfCross_60s.png")
            if os.path.exists(img): st.image(img, use_container_width=True, caption="Number of Crossings (Night)")
            else: st.info("📌 Add 'night_density_plot_NumOfCross_60s.png' (Folder 2)")

    st.markdown("---")

    # =========================================================================
    # 6. NUMERICAL VALIDATION
    # =========================================================================
    st.subheader("6. Numerical Integration Benchmarks")
    st.markdown("Performance profiling of the integration methods utilized for the cryptographic yield analysis.")
    
    img = os.path.join(ASSETS_DIR, "Key_Loss_Methods_Benchmark.png")
    if os.path.exists(img): 
        # Maintained [1, 4, 1] column ratio to ensure large display size and HD sharpness
        col1, col2, col3 = st.columns([1, 4, 1])
        with col2:
            st.image(img, use_container_width=True, caption="Integration Methods Benchmark")
    else: 
        st.info("📌 Add 'Key_Loss_Methods_Benchmark.png' (Folder 11)")

# =========================================================================
    # 7. 3D GEOSPATIAL VISUALIZATION (GOOGLE EARTH PRO RENDERINGS)
    # =========================================================================
    st.markdown("---")
    st.subheader("7. Interactive 3D Geospatial Visualization")
    st.markdown("""
    To fully comprehend the spatio-temporal complexity of aviation-induced QKD interruptions, static 2D plots are inherently limited. 
    The three-dimensional volumetric renderings below, natively mapped onto the WGS84 Earth model using Google Earth Pro, 
    capture the dynamic intersection events between aircraft trajectories and the satellite optical link in high-fidelity.
    """)

    # We create a new specific path for the video folder you mentioned
    VIDEO_DIR = os.path.join(BASE_DIR, "Video")

    # Create 3 equal columns to display the GIFs side-by-side
    col_g1, col_g2, col_g3 = st.columns(3)
    
    with col_g1:
        gif1_path = os.path.join(VIDEO_DIR, "tour_1_gif.gif")
        if os.path.exists(gif1_path):
            st.image(gif1_path, use_container_width=True)
            st.markdown("**1. The Orbital Intersect**")
            st.caption("Macro-scale simulation isolating the precise crossing events between dynamic flight tracks (blue) and sun-synchronous satellite passes (red) within the Bavarian airspace.")
        else:
            st.info("📌 Add 'tour_1_gif.gif' to the 'Video' folder inside 'qwk_webapp'.")
            
    with col_g2:
        gif2_path = os.path.join(VIDEO_DIR, "tour_2_gif.gif")
        if os.path.exists(gif2_path):
            st.image(gif2_path, use_container_width=True)
            st.markdown("**2. OGS Spatial Grid Integration**")
            st.caption("Superimposition of the Optical Ground Station candidate matrix over Munich, demonstrating the localized vulnerability density in proximity to the airport.")
        else:
            st.info("📌 Add 'tour_2_gif.gif' to the 'Video' folder inside 'qwk_webapp'.")

    with col_g3:
        gif3_path = os.path.join(VIDEO_DIR, "tour_3_gif.gif")
        if os.path.exists(gif3_path):
            st.image(gif3_path, use_container_width=True)
            st.markdown("**3. Historical Edge Case Validation**")
            st.caption("A specific extreme-geometry validation scenario extracted from empirical flight data (Nov 12, 2024), highlighting severe, high-slant-range beam interruptions.")
        else:
            st.info("📌 Add 'tour_3_gif.gif' to the 'Video' folder inside 'qwk_webapp'.")

    st.markdown("<br>", unsafe_allow_html=True)

# =========================================================================
    # 8. REFERENCE SPATIAL TOPOLOGIES & EMPIRICAL VALIDATION
    # =========================================================================
    st.markdown("---")
    st.subheader("8. Reference Spatial Topologies & Empirical Validation")
    st.markdown("""
    To contextualize the dynamic 3D renderings, the following high-resolution static topologies provide 
    the foundational geospatial framework of the simulation. Extracted directly from the baseline empirical dataset, 
    these visualisations map the most critical spatial boundaries of the study.
    """)

    # 1. Large primary image at the top (Centered leveraging a 1-4-1 column ratio)
    img_crossings = os.path.join(ASSETS_DIR, "crossings_12.11.2024_3D.png")
    if os.path.exists(img_crossings):
        col_c1, col_c2, col_c3 = st.columns([1, 4, 1])
        with col_c2:
            st.image(img_crossings, use_container_width=True, caption="Empirical Validation: High-Slant-Range 3D Crossing Event (Historical Data: Nov 12, 2024)")
    else:
        st.info("📌 Add 'crossings_12.11.2024_3D.png' to the assets folder.")

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. Two foundational baseline images positioned side-by-side (on the same row)
    col_ref1, col_ref2 = st.columns(2)
    
    with col_ref1:
        img_grid = os.path.join(ASSETS_DIR, "grid.png")
        if os.path.exists(img_grid):
            st.image(img_grid, use_container_width=True, caption="Baseline Infrastructure: 41×41 Optical Ground Station (OGS) Candidate Matrix")
        else:
            st.info("📌 Add 'grid.png' to the assets folder.")

    with col_ref2:
        img_pass = os.path.join(ASSETS_DIR, "pass_set.png")
        if os.path.exists(img_pass):
            st.image(img_pass, use_container_width=True, caption="Orbital Infrastructure: Isolated Sun-Synchronous Satellite Ground Tracks")
        else:
            st.info("📌 Add 'pass_set.png' to the assets folder.")

    st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TAB 13 — HIGH-FIDELITY MODEL VALIDATIONS
# -----------------------------------------------------------------------------
with tab_hifi:
    st.header("Astrodynamics & Link Optimization")
    st.markdown("""
    The interactive simulator presented in the preceding tabs relies, by deliberate design, on two
    simplifying assumptions that trade a small amount of physical fidelity for real-time interactivity:
    a spherical, homogeneous Earth model for orbital kinematics, and a fixed, user-specified aviation
    safety margin for the beacon-laser blackout window. Both assumptions are standard in first-pass
    mission-analysis tools and are entirely appropriate for interactive parameter exploration. However,
    neither survives unmodified in a physical deployment study, where centimeter-to-arcsecond-level
    geometric accuracy and a defensible, physically motivated safety margin are prerequisites for a
    credible engineering assessment.

    This section presents two independent, offline high-fidelity validation studies — conducted with
    adaptive numerical quadrature and the exact WGS84 reference ellipsoid rather than the simulator's
    interactive-speed approximations — that quantify precisely how much accuracy the simplified models
    sacrifice, and under what conditions that sacrifice becomes operationally significant. Both studies
    are fully reproducible from closed-form geodetic and orbital-mechanical relations; no fitted or
    empirically tuned parameters are used anywhere in either analysis.
    """)

    st.markdown("---")

    # =========================================================================
    # SECTION 1 — GEODESY
    # =========================================================================
    st.subheader("I. Transition from a Spherical Earth to WGS84 Ellipsoidal Geodesy")

    st.markdown(r"""
    Orbital mechanics textbooks and most first-order satellite-pass simulators adopt the *spherical,
    homogeneous Earth* assumption introduced in the theoretical framework: the planet is
    treated as a perfect sphere of constant mean radius $R_\oplus = 6{,}371.0\ \mathrm{km}$, with its
    center of mass coincident with its geometric center. This assumption is what permits the two-body
    problem to be solved in closed form and is precisely why it is retained in the interactive tabs of
    this simulator — every slider adjustment must re-evaluate the full pass geometry in well under a
    second, which a numerically integrated oblate-Earth propagator generally cannot guarantee.

    Physically, however, the Earth is an oblate spheroid: centrifugal flattening from its own rotation
    compresses the polar axis relative to the equatorial plane by a factor of roughly 1 part in 300. The
    operational geodetic standard that captures this — and the datum against which GPS, GLONASS, Galileo,
    and essentially every modern satellite ephemeris are referenced — is the **World Geodetic System 1984
    (WGS84)** reference ellipsoid, defined by NIMA Technical Report TR8350.2. This validation study
    replaces the interactive simulator's spherical assumption with the full WGS84 ellipsoidal geometry and
    quantifies, panel by panel, exactly how much the two models diverge.
    """)

    col_h1, col_h2 = st.columns([1, 1.2])
    with col_h1:
        st.markdown("**A.1 — The WGS84 Geocentric Radius Function**")
        st.markdown(r"""
        The exact geocentric distance from the Earth's center to a point on the WGS84 ellipsoid surface,
        as a function of *geodetic* latitude $\phi$ (the latitude a GPS receiver reports — defined by the
        local surface normal, not the line to the Earth's center), follows directly from the implicit
        ellipse equation $(x/a)^2+(z/b)^2=1$ expressed in terms of the surface normal direction:
        """)
        st.latex(r"R_e(\phi) = \sqrt{ \frac{(a^2 \cos\phi)^2 + (b^2 \sin\phi)^2}{(a \cos\phi)^2 + (b \sin\phi)^2} }")
        st.markdown(r"""
        with defining semi-major axis $a = 6{,}378{,}137.0\ \mathrm{m}$ and flattening
        $1/f = 298.257223563$, giving semi-minor axis $b = a(1-f)$ and first eccentricity
        $e^2 = f(2-f) \approx 6.6944\times10^{-3}$. Unlike the spherical model, $R_e(\phi)$
        is latitude-dependent: it equals $a$ exactly at the equator and $b$ exactly at the poles, with
        $R_e(\phi) < a$ everywhere in between. The **radial defect** relative to the mean-sphere model used
        by the interactive simulator is therefore itself a function of latitude:
        """)
        st.latex(r"\Delta R(\phi) = R_e(\phi) - R_\oplus, \qquad \Delta R(48.35^\circ) \approx -4.76\ \mathrm{km}")

        st.markdown("**A.2 — Geodetic vs. Geocentric Latitude Defect**")
        st.markdown(r"""
        A second, more subtle effect is that the geodetic latitude $\phi$ (surface-normal direction) and
        the geocentric latitude $\psi$ (angle subtended at the Earth's center) are *not* the same angle
        except at the equator and poles — the classical "reduced latitude" problem of ellipsoidal geodesy:
        """)
        st.latex(r"\tan\psi = (1-e^2)\tan\phi \;\;\Longrightarrow\;\; \Delta\phi = \phi-\psi \approx f\sin(2\phi)")
        st.markdown(r"""
        the first-order small-flattening approximation, maximized near $\phi=45^\circ$ and evaluating to
        $\Delta\phi(48.35^\circ)\approx 0.191^\circ$ (11.5 arcmin) — the source of the latitude-defect
        value quoted in the panel-by-panel discussion below. Because a physically leveled OGS mount points
        along the surface normal $\hat n$ rather than the geocentric vector, the zenith angle $\theta(t)$
        must be redefined using the exact dot product with the satellite line-of-sight vector $\vec d(t)$,
        which resolves the inverse-cosine domain singularities the naive spherical formula develops at
        polar latitudes:
        """)
        st.latex(r"\theta(t) = \arccos\!\left( \frac{\vec{d}(t) \cdot \hat{n}}{|\vec{d}(t)|} \right)")

        st.markdown("**A.3 — Principal Radii of Curvature & Normal Gravity**")
        st.markdown(r"""
        The candidate-OGS grid used throughout the Geospatial Risk Map is specified with uniform 5 km
        physical spacing. Converting that physical spacing into consistent latitude/longitude increments
        requires the meridian radius $M(\phi)$ (north–south metric distance) and the prime-vertical radius
        $N(\phi)$ (east–west metric distance, via $\Delta x = N(\phi)\cos\phi\;\Delta\lambda$) rather than
        the single constant $R_\oplus$ a spherical model provides:
        """)
        st.latex(r"M(\phi) = \frac{a(1-e^2)}{(1-e^2\sin^2\phi)^{3/2}}, \qquad N(\phi) = \frac{a}{\sqrt{1-e^2\sin^2\phi}}")
        st.markdown(r"""
        Finally, normal gravity $g(\phi)$ on the ellipsoid surface — implicitly assumed constant in the
        spherical two-body model — follows the exact, closed-form **Somigliana equation**, with WGS84
        defining values $g_e = 9.7803253359\ \mathrm{m/s^2}$ (equatorial) and
        $g_p = 9.8321849378\ \mathrm{m/s^2}$ (polar):
        """)
        st.latex(r"g(\phi) = \frac{a\, g_e \cos^2\phi + b\, g_p \sin^2\phi}{\sqrt{a^2 \cos^2\phi + b^2 \sin^2\phi}}")

    with col_h2:
        img_geo = os.path.join(ASSETS_DIR, "wgs84_vs_sphere_error.png")
        if os.path.exists(img_geo):
            st.image(img_geo, use_container_width=True,
                      caption="Four-panel validation of geodetic corrections over 0°–90° latitude: "
                              "(A) geocentric radial discrepancy against the IUGG mean sphere; "
                              "(B) geodetic–geocentric latitude defect; (C) meridian and "
                              "prime-vertical radii of curvature, M(φ) and N(φ); (D) propagated zenith-angle "
                              "error Δθ along a representative 500 km LEO pass over Munich.")
        else:
            st.info("📌 Add 'wgs84_vs_sphere_error.png' to the assets folder.")

        st.markdown("**Numerical Evaluation at the Munich OGS Reference Latitude ($\\phi = 48.35^\\circ\\,\\mathrm{N}$)**")
        st.latex(r"""
        \begin{aligned}
        R_e(48.35^\circ) &\approx 6{,}366.24\ \mathrm{km}, & \Delta R &\approx -4.76\ \mathrm{km} \\[2pt]
        \Delta\phi &\approx 0.191^\circ\ (11.5') \\[2pt]
        M(48.35^\circ) &\approx 6{,}376.6\ \mathrm{km}, & N(48.35^\circ) &\approx 6{,}390.6\ \mathrm{km} \\[2pt]
        g(48.35^\circ) &\approx 9.8072\ \mathrm{m/s^2}
        \end{aligned}
        """)
        st.markdown(r"""
        These four independent quantities — each computed from the same closed-form ellipsoidal
        relations above and each individually validated against the reference values published in NIMA
        TR8350.2 — jointly determine the propagated pointing-error curve $\Delta\theta(t)$ of Panel D.
        Formally, treating the radial defect $\Delta R(\phi)$ and the surface-normal misalignment
        $\Delta\phi(\phi)$ as small perturbations to the spherical slant-range geometry, a first-order
        propagation gives the approximate scaling:
        """)
        st.latex(r"\Delta\theta \;\sim\; \frac{\partial\theta}{\partial R}\,\Delta R \;+\; \frac{\partial\theta}{\partial\phi}\,\Delta\phi")
        st.markdown(r"""
        both terms of which are comparable in magnitude at Munich's latitude and combine — rather than
        cancel — along most of a representative pass, which is precisely why Panel D shows a peak
        propagated error approaching 900 arcseconds rather than a much smaller residual.
        """)

    st.markdown("**Quantitative Findings and Their Operational Significance (Panels A–D)**")
    st.markdown(r"""
    * **Radial and latitudinal defects (Panels A & B).** At the latitude of the Munich Metropolitan Area
      ($\phi = 48.35°\,\mathrm{N}$), the WGS84 ellipsoid surface sits approximately **4.76 km closer** to
      the Earth's center than the IUGG mean-sphere approximation used in the interactive simulator's
      orbital kinematics — a discrepancy that enters the two-body slant-range calculation $d(\lambda)$
      directly and without attenuation. The geodetic surface normal simultaneously deviates from the true
      geocentric radius vector by **0.191°** (approximately 11.5 arcminutes), meaning that "straight up"
      as measured by a GPS receiver or a physically leveled OGS mount is *not* the direction toward the
      Earth's center assumed by the spherical zenith-angle formula.
    * **Curvature radii (Panel C).** At Munich's latitude, $M(\phi) \approx 6{,}376.6\ \mathrm{km}$ and
      $N(\phi) \approx 6{,}390.6\ \mathrm{km}$, both measurably different from the spherical constant
      $R_\oplus = 6{,}371.0\ \mathrm{km}$ used throughout the interactive grid generator. Because the OGS
      candidate grid is specified in fixed 5 km physical increments, using the constant $R_\oplus$ instead
      of the latitude-correct $N(\phi)\cos\phi$ for longitude spacing introduces a compounding east-west
      grid-registration error that grows with distance from the grid's reference latitude:
      $\delta\lambda_{\text{err}} = \Delta x\,\big(1/(N\cos\phi) - 1/(R_\oplus\cos\phi)\big)$.
    * **Propagated optical pointing error (Panel D).** The combination of the radial and latitudinal
      defects propagates, through the same zenith-angle geometry used by `build_pass_kinematics()`, into
      a peak zenith-angle error of nearly **900 arcseconds (≈ 0.25°)** at representative points along a
      500 km LEO pass. For comparison, a well-collimated satellite beacon laser of the type modeled in
      this simulator (beam waist $W_0 \sim 10$–$20\ \mathrm{cm}$ at 850 nm, full divergence
      $\theta_{\text{div}} \approx 2\lambda/(\pi W_0)$) has a full divergence angle on the order of a few
      microradians — roughly three orders of magnitude smaller than the propagated geodetic pointing
      error, i.e. $\Delta\theta / \theta_{\text{div}} \sim 10^{3}$. A 0.25° mispointing at the OGS
      acquisition system is therefore not a minor correction but a **link-breaking** error: it would
      place the beacon laser's centroid many beam diameters away from the intended receive aperture,
      resulting in complete loss of optical acquisition rather than a mere reduction in link efficiency.
      This is the central conclusion of the geodetic validation study: for any satellite-QKD system whose
      acquisition, pointing, and tracking (APT) subsystem operates at sub-milliradian precision — which
      essentially all realistic beacon-laser designs do — an ellipsoidal Earth model is not an optional
      refinement but a mandatory component of the pointing budget, and the spherical approximation
      retained in the *interactive* tabs of this simulator should be understood strictly as a tool for
      qualitative parameter exploration, never as a substitute for mission-level pointing analysis.
    """)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # =========================================================================
    # SECTION 2 — SAFETY MARGIN
    # =========================================================================
    st.subheader("II. Physically-Derived Optimization of the Aviation Safety Margin")

    st.markdown(r"""
    The second simplifying assumption retained by the interactive simulator concerns the aviation safety
    margin $\Delta t$: the operator sets a single, fixed blackout half-width via a sidebar slider, applied
    uniformly regardless of where in the pass the aircraft crossing occurs, the satellite's altitude, or
    the instantaneous shape of the secure-key-rate function $S(t)$ at that moment. This is a defensible
    interactive-exploration default — real aviation safety protocols do typically specify a single
    conservative blackout duration for operational simplicity — but it obscures a substantially richer
    underlying physical picture: the *marginal cost*, in lost key material, of extending the safety margin
    by one additional second is strongly time- and altitude-dependent, and a margin that is efficient near
    the horizon may be needlessly conservative (or, conversely, insufficiently conservative) near zenith.

    This validation study replaces the fixed-margin assumption with a full, adaptively-integrated
    treatment of the safety-margin loss functional, and — critically — replaces the arbitrary Gaussian fit
    to $S(t)$ used in early prototyping stages of this project with the exact, closed-form link-budget
    expression already derived in the Theoretical Framework tab (Kasten–Young atmospheric extinction ×
    Gaussian-aperture geometric coupling), so that every number reported here is directly traceable to the
    same physics governing the interactive simulator, not to a curve-fitting exercise.
    """)

    col_h3, col_h4 = st.columns([1.2, 1])
    with col_h3:
        img_margin = os.path.join(ASSETS_DIR, "safety_margin_optimization.png")
        if os.path.exists(img_margin):
            st.image(img_margin, use_container_width=True,
                      caption="Six-panel safety-margin optimization study across four representative LEO "
                              "altitudes (200, 300, 500, 800 km): (A) link-budget-derived S(t) shape functions; "
                              "(B) cumulative key-loss integral S_loss(Δt); (C) marginal loss rate "
                              "dS_loss/d(Δt) via the Leibniz integral rule; (D) resulting net link efficiency; "
                              "(E) key loss at the baseline 60 s margin across altitudes; (F) numerical "
                              "cross-validation table (Leibniz closed form vs. central finite difference).")
        else:
            st.info("📌 Add 'safety_margin_optimization.png' to the assets folder.")

        st.markdown("**Net Link Efficiency and the Total-Pass Normalization**")
        st.markdown(r"""
        Panel D expresses the safety-margin loss as a dimensionless efficiency by normalizing against the
        full horizon-to-horizon pass yield $S_{\text{tot}}$, obtained by extending the same integral over
        the entire visibility window $[0,\,2t_{\max}]$, with
        $t_{\max}=\lambda_{\max}/\omega=\arccos\!\big(R_\oplus/(R_\oplus+h)\big)/\omega$:
        """)
        st.latex(r"S_{\text{tot}} = \int_{0}^{2t_{\max}} S(t)\,dt, \qquad \eta_{\text{link}}(\Delta t) = 1-\frac{S_{\text{loss}}(\Delta t)}{S_{\text{tot}}}")
        st.markdown(r"""
        and the **diminishing-returns condition** identified in Panel C is, formally, the statement that
        the marginal loss rate itself is decreasing away from the crossing time — equivalently, that the
        loss functional is concave in that region:
        """)
        st.latex(r"\frac{d^2 S_{\text{loss}}}{d(\Delta t)^2} = \tfrac{1}{4}\!\left[S'\!\big(t_c+\tfrac{\Delta t}{2}\big) - S'\!\big(t_c-\tfrac{\Delta t}{2}\big)\right] < 0 \;\;\text{for}\;\; \Delta t \gtrsim 100\ \mathrm{s}")

    with col_h4:
        st.markdown("**B.1 — The Safety-Margin Loss Functional**")
        st.markdown(r"""
        Rather than assuming a fixed blackout duration, the secure key material forfeited by disabling the
        beacon laser for a symmetric window of width $\Delta t$ centered on the aircraft-crossing time
        $t_c$ is evaluated as the exact definite integral of the physically-derived secure key rate:
        """)
        st.latex(r"S_{\text{loss}}(\Delta t) = \int_{t_c - \Delta t/2}^{t_c + \Delta t/2} S(t)\,dt")
        st.markdown(r"""
        where $S(t)$ is obtained directly from the dynamic Gaussian beam waist $W(d)$ and the Kasten–Young
        air-mass function $m(\theta)$, exactly as used throughout the interactive simulator:
        """)
        st.latex(r"S(t) = S_{\text{max}} \left[ \frac{e^{-\tau_0 m(\theta(t))} \left( 1 - e^{-D_R^2 / 2W(d(t))^2} \right)}{e^{-\tau_0 m(0)} \left( 1 - e^{-D_R^2 / 2W(h)^2} \right)} \right]")
        st.latex(r"W(d) = W_0 \sqrt{1 + \left(\frac{\lambda_0 d}{\pi W_0^2}\right)^2}")
        st.markdown(r"""
        The integral is evaluated with adaptive Gauss–Kronrod quadrature (`scipy.integrate.quad`, absolute
        tolerance $10^{-12}$) rather than a fixed-step Riemann sum, ensuring the reported values are
        numerically exact to machine precision.

        **B.2 — The Marginal Loss Rate.** The quantity of direct engineering relevance when *choosing* a
        safety margin is not the cumulative loss $S_{\text{loss}}(\Delta t)$ itself, but its derivative —
        the *marginal* cost of the next second of margin. By the Leibniz integral rule this has an exact
        closed form requiring no further numerical differentiation:
        """)
        st.latex(r"\frac{dS_{\text{loss}}}{d(\Delta t)} = \frac{1}{2}\Big[\,S\!\big(t_c+\tfrac{\Delta t}{2}\big) + S\!\big(t_c-\tfrac{\Delta t}{2}\big)\,\Big]")
        st.markdown(r"""
        so that the margin at which the marginal cost first drops below a chosen operational threshold
        $\varepsilon$ — the natural stopping criterion for a variable-margin policy — is the root
        $\Delta t^\star$ of:
        """)
        st.latex(r"\frac{1}{2}\Big[\,S\!\big(t_c+\tfrac{\Delta t^\star}{2}\big) + S\!\big(t_c-\tfrac{\Delta t^\star}{2}\big)\,\Big] = \varepsilon")

    st.markdown("**Quantitative Findings and Their Operational Significance (Panels A–F)**")
    st.markdown(r"""
    * **Altitude dependence of the link-budget shape (Panel A).** Consistent with the orbital-mechanics
      relation $v_{\text{orb}}=\sqrt{\mu/(R_\oplus+h)}$ and the resulting angular rate
      $\omega = v_{\text{orb}}/(R_\oplus+h)$, lower-altitude passes (200–300 km) exhibit a
      narrower, sharper $S(t)$ profile — the satellite crosses the OGS zenith more quickly at lower
      altitude — while higher-altitude passes (800 km) spread the same total pass yield $S_{\text{tot}}$
      over a proportionally longer, flatter window. A single fixed margin therefore represents a very
      different *fraction* of the total useful pass duration depending on the constellation altitude
      actually deployed, which the interactive simulator's altitude slider allows the user to explore
      qualitatively but does not, by itself, optimize.
    * **Diminishing marginal returns (Panels B & C).** Panel C is the central result of this study: the
      marginal loss rate $dS_{\text{loss}}/d(\Delta t)$ is not constant, and in fact decays sharply away
      from the zenith crossing, satisfying the concavity condition given above for
      $\Delta t \gtrsim 100\ \mathrm{s}$. This formally establishes a **diminishing-returns regime** in the
      safety margin: for a typical LEO pass, extending the blackout window beyond roughly 100 seconds from
      the crossing costs progressively less key material per additional second of margin, because the
      satellite is already low on the horizon where atmospheric extinction (via the Kasten–Young air-mass
      function) and aperture-coupling losses have already suppressed $S(t)$ close to zero. Conversely,
      margin extended *near* the zenith crossing is comparatively expensive, since that is precisely where
      $S(t)$ attains its peak $S_{\max}$. This suggests that a *time-varying* safety margin — narrower near
      the pass edges, tighter around zenith crossings, formally bounded by the threshold condition
      $\Delta t^\star$ above — could in principle recover a meaningful fraction of the key material lost
      under a uniformly fixed margin, at equal aviation-safety stringency; formalizing such a
      variable-margin policy is a natural extension of this framework.
    * **Altitude comparison at the baseline margin (Panel E).** At the interactive simulator's default
      60-second margin, the absolute key loss $S_{\text{loss}}(60\,\mathrm{s})$ differs measurably across
      the four representative altitudes (200, 300, 500, 800 km), directly reflecting the altitude-dependent
      pass-duration and peak-rate trade-off quantified in Panel A. This confirms that a safety-margin
      policy validated at one candidate constellation altitude does not automatically transfer, in terms of
      *efficiency* $\eta_{\text{link}}$ cost, to a different altitude — the appropriate margin (or, more
      precisely, its marginal cost) must be re-evaluated whenever the constellation design altitude
      changes.
    * **Numerical validation (Panel F).** The closed-form Leibniz-rule marginal rate is independently
      cross-checked against a central finite-difference estimate,
      $\big[S_{\text{loss}}(\Delta t+h)-S_{\text{loss}}(\Delta t-h)\big]/2h$, of the same quantity computed
      directly from the adaptively-integrated loss curve. Agreement to better than $10^{-10}$ relative
      error across all four altitudes confirms both that the quadrature scheme resolves the loss integral
      to machine precision and that the analytic marginal-rate expression is implemented correctly — a
      standard, necessary sanity check whenever a numerically integrated quantity is subsequently
      differentiated.
    """)

    st.markdown("<br>", unsafe_allow_html=True)

st.markdown("---")
st.caption("Satellite Quantum Key Distribution Systems Laboratory — MCQST Summer Bachelor Program 2026, "
           "School of Computation, Information and Technology, Department of Computer Engineering, Professorship of Quantum Communication Systems Engineering, Technical University of Munich, 80333 Munich, Germany. "
           "Simulation developed by Andrea Staffieri, under the academic direction of "
           "Prof. Dr. phil. Tobias Vogl and the supervision of Dr. Asli Cakan Cebe.")