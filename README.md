# Simulation and Evaluation of Satellite-Based Quantum Key Distribution Links Under Aviation-Constrained Optical Ground Station Operation 🛰️🔑

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)
![Institution](https://img.shields.io/badge/Institution-TUM-0065BD.svg)
![Program](https://img.shields.io/badge/Program-MCQST_2026-black.svg)

A physically self-consistent digital twin of a satellite-to-ground BB84 Quantum Key Distribution (QKD) link. This application provides a full-stack simulation environment integrating two-body orbital kinematics, free-space optical link budgets, decoy-state security analysis, eavesdropping models, and geospatial vulnerability mapping due to aviation-induced beam occultation.

This project was developed within the **MCQST Summer Bachelor Program 2026** at the **Technical University of Munich (TUM)**, Professorship of Quantum Communication Systems Engineering.

---

## 📌 Project Overview

Satellite-based QKD offers a pathway to global quantum-secure communication. However, Optical Ground Stations (OGS) located near major metropolitan areas (such as Munich) face frequent line-of-sight interruptions due to dense commercial air traffic. To comply with aviation safety regulations, beacon lasers must be shuttered during aircraft crossings, leading to critical key-generation blackouts.

This digital twin allows researchers and engineers to:
1. **Propagate LEO satellite passes** using realistic astrodynamics and slant-range geometries.
2. **Compute dynamic quantum link budgets** incorporating Kasten-Young atmospheric extinction and Gaussian beam divergence.
3. **Analyze cryptographic security** under the BB84 vacuum + weak decoy-state protocol against advanced eavesdropping (Intercept-Resend & PNS attacks).
4. **Quantify aviation-induced key loss** using deterministic spatio-temporal intersection algorithms.
5. **Optimize OGS placement** across the Munich Metropolitan Area to minimize safety-blackout disruptions.

---

## 🚀 Key Features

* **Theoretical Framework Engine:** Built-in documentation detailing the physics of quantized single-mode fields, relativistic time-dilation offsets, and the Pirandola-Laurenza-Ottaviani-Banchi (PLOB) repeaterless bound.
* **Live Mission Control:** Interactive playback of the satellite pass, calculating instantaneous zenith angles, Quantum Bit Error Rate (QBER), and secure key fractions in real-time.
* **High-Fidelity Optical Occultation:** 3D surface modeling of Gaussian beam intensity drops caused by aircraft cross-sections intersecting the optical axis.
* **Geospatial Risk Mapping:** Empirical validation grids plotting candidate OGS locations against real-world Munich/Ingolstadt flight trajectory density.
* **Offline High-Fidelity Validation Archives:** Access to deep-dive geodetic (WGS84 vs. Spherical Earth) and adaptive numerical quadrature validation runs.

---

## ⚙️ Installation & Setup

To run this simulation locally, ensure you have Python 3.9+ installed.
