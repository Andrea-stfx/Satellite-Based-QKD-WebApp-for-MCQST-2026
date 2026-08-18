# Simulation and Evaluation of Satellite-Based Quantum Key Distribution Links and Optical Ground Station Locations Near Airports 🛰️🔑

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)
![Institution](https://img.shields.io/badge/Institution-TUM-0065BD.svg)
![Program](https://img.shields.io/badge/Program-MCQST_2026-black.svg)

**Authors:** Andrea Staffieri<sup>1,2,3</sup>, Asli Çakan Cebe<sup>2,3</sup>, Tobias Vogl<sup>2,3</sup>

<sup>1</sup> <i>Department of Electrical and Information Engineering (DEI), Polytechnic University of Bari, Italy.</i>  
<sup>2</sup> <i>School of Computation, Information and Technology, Department of Computer Engineering, Technical University of Munich, Germany.</i>  
<sup>3</sup> <i>Munich Center for Quantum Science and Technology (MCQST), Germany.</i>

---

## 📌 Abstract & Project Objective

Satellite-based Quantum Key Distribution (QKD) represents an emerging technology capable of providing unconditionally secure global communication. Establishing a stable free-space optical link between a Low Earth Orbit (LEO) satellite and an Optical Ground Station (OGS) inherently requires a strictly unobstructed line-of-sight. 

A critical operational constraint arises in regions characterized by dense airspace, such as the Munich Metropolitan Area. Since satellite tracking relies on high-intensity beacon lasers, strict aviation safety regulations mandate the forceful deactivation of the laser whenever an aircraft intersects the optical path. This mandatory shutdown results in a temporary loss of beam alignment, effectively severing the quantum channel and causing an irrecoverable loss of cryptographic key material.

**Objective:** To engineer a comprehensive simulation framework capable of evaluating candidate OGS locations by mathematically detecting aircraft-satellite trajectory crossings and minimizing aviation-induced cryptographic losses.

---

## 🚀 System Architecture and Data Pipeline

To rigorously evaluate the spatial vulnerability of different locations, a deterministic simulation pipeline combining real-world aviation telemetry and orbital mechanics was constructed. This multi-stage approach ensures both static geographic constraints and dynamic traffic patterns are accounted for.

*   **The Geospatial Matrix:** A uniform grid of candidate OGS locations was generated across the Munich region utilizing a 5 km spacing. This was initially referenced to a spherical Earth model, serving as the foundational framework for subsequent high-fidelity geodetic calibrations.
*   **Aviation Telemetry:** Real-world flight trajectories, encompassing both arrivals and departures for Munich and Ingolstadt airports, were acquired, cleaned, and integrated directly into the simulation space.
*   **Orbital Propagation:** A dataset comprising 51 distinct LEO satellite passes was generated, assuming a sun-synchronous orbit at an altitude of 500 km and a 98° inclination. The Right Ascension of the Ascending Node (RAAN) was varied systematically in 1° steps to ensure an unbiased probability distribution of overhead passes.

---

## 🧮 Crossing Detection and Cryptographic Loss

The core of the simulation relies on detecting 3D spatio-temporal intersections between aircraft trajectories and the satellite's line-of-sight vector. 

*   **Intersection Mathematics:** For each candidate OGS, the algorithm formulates a Line Segment Intersection problem between the flight vector $CD$ and the instantaneous satellite-to-OGS vector $AB$. This is solved via Cramer's rule to determine the exact epoch $t_c$ of the aircraft crossing.
*   **The Loss Functional:** A safety margin of duration $\Delta t$ is enforced symmetrically around the crossing time $t_c$. The cryptographic key forfeited during this blackout is computed by integrating the Secure Key Rate $S(t)$ over this interval:

$$S_{\text{loss}}=\int_{t_c-\Delta t}^{t_c+\Delta t}S(t)dt$$

---

## 🌍 Geospatial Vulnerability Mapping

By calculating $S_{\text{loss}}$ for every crossing across all generated passes, the aviation-induced vulnerabilities across the Bavarian airspace were mapped.

*   **Density Analysis:** The aggregated key loss was visualized using high-resolution spatial density plots.
*   **Operational Insights:** The results highlight critical "red zones" that are strictly correlated with the approach and climb-out glide slopes of Munich Airport. OGS locations situated directly beneath these corridors suffer from severe communication degradation due to repeated safety blackout enforcement.
*   **Kinematic Vulnerability Scaling:** Evaluating aircraft speed-versus-altitude profiles across variable satellite altitudes (200, 300, 500, and 800 km) reveals a dynamic intersection threshold.

---

## 🔬 High-fidelity Refinements and Optimization

### I. Transition to WGS84 Ellipsoidal Geodesy
Assuming a perfectly spherical Earth introduces severe geometric biases in laser targeting. The spatial engine was therefore upgraded to the exact WGS84 reference ellipsoid. 
*   **Result:** At Munich's latitude (48.35°N), the true surface is 4.76 km closer to the geocenter than a mean sphere. This radial defect, combined with the geocentric latitude offset of 0.191°, propagates into an optical pointing error peaking at 900 arcseconds (0.25°). For a quantum beacon laser with microradian divergence, this error causes a total target miss, proving that ellipsoidal modeling is a mandatory requirement.

### II. Dynamic Safety Margin Optimization
Standard protocols assume a static blackout duration. Using the Leibniz integral rule, the exact Marginal Loss Rate $dS_{\text{loss}}/d(\Delta t)$ was calculated.
*   **Result:** The mathematical evaluation proves a diminishing-returns regime. Extending a blackout window beyond approximately 100 s incurs negligible additional key loss, as atmospheric attenuation near the horizon already suppresses the channel capacity to zero.

---

## 🔮 Conclusions and Future Developments

The developed simulation framework successfully detects aircraft-satellite crossings and quantifies their impact on QKD efficiency. By integrating real-world telemetry with rigorous WGS84 ellipsoidal geodesy and dynamically optimized safety margins, the model provides network architects with an exact mathematical tool to optimize OGS placement in congested airspaces.

**Future Work:**
*   Integration of dynamic weather models and atmospheric turbulence profiling.
*   Implementation of real-time, flight-aware variable safety margins based on aircraft altitude and speed.

---

## 🤝 Acknowledgements
This research was conducted as part of the MCQST Summer Bachelor Program 2026 at the Technical University of Munich (TUM), Department of Computer Engineering. Special gratitude is expressed to the project supervisors, Prof. Dr. phil. Tobias Vogl and Dr. Asli Cakan Cebe, for their invaluable guidance, support, and academic direction throughout this project.
