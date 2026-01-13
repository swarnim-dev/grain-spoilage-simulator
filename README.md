# Grain Spoilage Simulator 🌾

A physics-based grain storage simulator that models heat and moisture diffusion inside stored grain and estimates spoilage risk using probabilistic AI.

This project was built for a GDG hackathon and focuses on combining **engineering fundamentals** with **AI-driven uncertainty modeling**.

---

## 🧠 What This Project Does

- Simulates **temperature diffusion** inside a grain bulk
- Simulates **moisture migration** influenced by thermal gradients
- Models **biological spoilage** through dry matter loss (DML)
- Accounts for:
  - Different **crop types** (wheat, rice, maize)
  - **Silo wall thickness** (insulation effects)
- Uses **Google TensorFlow Probability** to generate a **probability distribution** of spoilage risk instead of a single fixed value
- Provides an interactive **GUI** for experimentation and visualization

---

## 🔬 Modeling Approach

### Physics-Based Simulation
The system solves a coupled set of equations:
- Heat diffusion PDE (Fourier’s law)
- Moisture diffusion PDE
- Nonlinear spoilage kinetics ODE

These equations are solved numerically using the **finite difference method** with explicit time stepping.

### Probabilistic AI Layer
Simulation outputs (dry matter loss) are post-processed using a **Bayesian probability model** implemented with **TensorFlow Probability**, allowing uncertainty-aware spoilage risk estimation.

---

## 🖥️ Application Interface

The GUI allows users to:
- Select crop type
- Set base temperature and hotspot temperature
- Define silo wall thickness
- Visualize:
  - Temperature profile
  - Moisture profile
  - Spoilage risk probability distribution

A CSV report is automatically generated after each simulation.

---

## Getting Started

### Prerequisites
- Python **3.10 or 3.11** (TensorFlow is not stable on Python 3.13)
- macOS / Linux / Windows

### Installation
```bash
pip install numpy matplotlib tensorflow tensorflow-probability
