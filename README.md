# Simulation and Evaluation of Satellite-Based Quantum Key Distribution Links and Optical Ground Station Locations Near Airports 🛰️🔑

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)
![Institution](https://img.shields.io/badge/Institution-TUM-0065BD.svg)
![Program](https://img.shields.io/badge/Program-MCQST_2026-black.svg)

**Authors:** Andrea Staffieri$^{1,2,3}$, Asli Çakan Cebe$^{2,3}$, Tobias Vogl$^{2,3}$[cite: 1].
$^{1}$ *Department of Electrical and Information Engineering (DEI), Polytechnic University of Bari, Italy*[cite: 1].
$^{2}$ *School of Computation, Information and Technology, Department of Computer Engineering, Technical University of Munich, Germany*[cite: 1].
$^{3}$ *Munich Center for Quantum Science and Technology (MCQST), Germany*[cite: 1].

---

## 📌 Abstract & Project Objective

Satellite-based Quantum Key Distribution (QKD) represents an emerging technology capable of providing unconditionally secure global communication[cite: 1]. Establishing a stable free-space optical link between a Low Earth Orbit (LEO) satellite and an Optical Ground Station (OGS) inherently requires a strictly unobstructed line-of-sight[cite: 1]. 

A critical operational constraint arises in regions characterized by dense airspace, such as the Munich Metropolitan Area[cite: 1]. Since satellite tracking relies on high-intensity beacon lasers, strict aviation safety regulations mandate the forceful deactivation of the laser whenever an aircraft intersects the optical path[cite: 1]. This mandatory shutdown results in a temporary loss of beam alignment, effectively severing the quantum channel and causing an irrecoverable loss of cryptographic key material[cite: 1].

**Objective:** To engineer a comprehensive simulation framework capable of evaluating candidate OGS locations by mathematically detecting aircraft-satellite trajectory crossings and minimizing aviation-induced cryptographic losses[cite: 1].

---

## 🚀 System Architecture and Data Pipeline

To rigorously evaluate the spatial vulnerability of different locations, a deterministic simulation pipeline combining real-world aviation telemetry and orbital mechanics was constructed[cite: 1]. This multi-stage approach ensures both static geographic constraints and dynamic traffic patterns are accounted for[cite: 1].

*   **The Geospatial Matrix:** A uniform grid of candidate OGS locations was generated across the Munich region utilizing a $5\text{ km}$ spacing[cite: 1]. This was initially referenced to a spherical Earth model, serving as the foundational framework for subsequent high-fidelity geodetic calibrations[cite: 1].
*   **Aviation Telemetry:** Real-world flight trajectories, encompassing both arrivals and departures for Munich and Ingolstadt airports, were acquired, cleaned, and integrated directly into the simulation space[cite: 1].
*   **Orbital Propagation:** A dataset comprising 51 distinct LEO satellite passes was generated, assuming a sun-synchronous orbit at an altitude of $500\text{ km}$ and a $98^\circ$ inclination[cite: 1]. The Right Ascension of the Ascending Node (RAAN) was varied systematically in $1^\circ$ steps to ensure an unbiased probability distribution of overhead passes[cite: 1].

---

## 🧮 Crossing Detection and Cryptographic Loss

The core of the simulation relies on detecting 3D spatio-temporal intersections between aircraft trajectories and the satellite's line-of-sight vector[cite: 1]. 

*   **Intersection Mathematics:** For each candidate OGS, the algorithm formulates a Line Segment Intersection problem between the flight vector $CD$ and the instantaneous satellite-to-OGS vector $AB$[cite: 1]. This is solved via Cramer's rule to determine the exact epoch $t_c$ of the aircraft crossing[cite: 1].
*   **The Loss Functional:** A safety margin of duration $\Delta t$ is enforced symmetrically around the crossing time $t_c$[cite: 1]. The cryptographic key forfeited during this blackout is computed by integrating the Secure Key Rate $S(t)$ over this interval[cite: 1]:

$$ S_{\text{loss}} = \int_{t_c-\Delta t}^{t_c+\Delta t} S(t) \, dt $$

---

## 🌍 Geospatial Vulnerability Mapping

By calculating $S_{\text{loss}}$ for every crossing across all generated passes, the aviation-induced vulnerabilities across the Bavarian airspace were mapped[cite: 1].

*   **Density Analysis:** The aggregated key loss was visualized using high-resolution spatial density plots[cite: 1].
*   **Operational Insights:** The results highlight critical "red zones" that are strictly correlated with the approach and climb-out glide slopes of Munich Airport[cite: 1]. OGS locations situated directly beneath these corridors suffer from severe communication degradation due to repeated safety blackout enforcement[cite: 1].
*   **Kinematic Vulnerability Scaling:** Evaluating aircraft speed-versus-altitude profiles across variable satellite altitudes ($200$, $300$, $500$, and $800\text{ km}$) reveals a dynamic intersection threshold[cite: 1].

---

## 🔬 High-fidelity Refinements and Optimization

### I. Transition to WGS84 Ellipsoidal Geodesy
Assuming a perfectly spherical Earth introduces severe geometric biases in laser targeting[cite: 1]. The spatial engine was therefore upgraded to the exact WGS84 reference ellipsoid[cite: 1]. 
*   **Result:** At Munich's latitude ($48.35^\circ\text{N}$), the true surface is $4.76\text{ km}$ closer to the geocenter than a mean sphere[cite: 1]. This radial defect, combined with the geocentric latitude offset of $0.191^\circ$, propagates into an optical pointing error peaking at $900\text{ arcseconds}$ ($0.25^\circ$)[cite: 1]. For a quantum beacon laser with microradian divergence, this error causes a total target miss, proving that ellipsoidal modeling is a mandatory requirement[cite: 1].

### II. Dynamic Safety Margin Optimization
Standard protocols assume a static blackout duration[cite: 1]. Using the Leibniz integral rule, the exact Marginal Loss Rate $dS_{\text{loss}}/d(\Delta t)$ was calculated[cite: 1].
*   **Result:** The mathematical evaluation proves a diminishing-returns regime[cite: 1]. Extending a blackout window beyond approximately $100\text{ s}$ incurs negligible additional key loss, as atmospheric attenuation near the horizon already suppresses the channel capacity to zero[cite: 1].

---

## 🔮 Conclusions and Future Developments

The developed simulation framework successfully detects aircraft-satellite crossings and quantifies their impact on QKD efficiency[cite: 1]. By integrating real-world telemetry with rigorous WGS84 ellipsoidal geodesy and dynamically optimized safety margins, the model provides network architects with an exact mathematical tool to optimize OGS placement in congested airspaces[cite: 1].

**Future Work:**
*   Integration of dynamic weather models and atmospheric turbulence profiling[cite: 1].
*   Implementation of real-time, flight-aware variable safety margins based on aircraft altitude and speed[cite: 1].

---

## 🤝 Acknowledgements
This research was conducted as part of the MCQST Summer Bachelor Program 2026 at the Technical University of Munich (TUM), Department of Computer Engineering[cite: 1]. Special gratitude is expressed to the project supervisors, Prof. Dr. phil. Tobias Vogl and Dr. Asli Cakan Cebe, for their invaluable guidance, support, and academic direction throughout this project[cite: 1].

## ⚙️ Installation & Setup

To run this simulation locally, ensure you have Python 3.9+ installed.
