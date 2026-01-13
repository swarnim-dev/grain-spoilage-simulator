import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import csv

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

plt.rcParams["figure.autolayout"] = False

CROPS = {
    "Wheat": {
        "rho": 780.0,
        "Cp": 1600.0,
        "k": 0.18,
        "Dm": 2e-8,
        "spoilage_factor": 1.0
    },
    "Rice": {
        "rho": 750.0,
        "Cp": 1500.0,
        "k": 0.14,
        "Dm": 3e-8,
        "spoilage_factor": 1.3
    },
    "Maize": {
        "rho": 720.0,
        "Cp": 1700.0,
        "k": 0.16,
        "Dm": 2.5e-8,
        "spoilage_factor": 1.1
    }
}

def run_simulation(days, T_base, T_hot, M_base, crop, wall_thickness):
    props = CROPS[crop]

    rho = props["rho"]
    Cp = props["Cp"]
    k_cond = props["k"]
    Dm = props["Dm"]
    spoilage_factor = props["spoilage_factor"]

    alpha = k_cond / (rho * Cp)
    delta = 1e-4

    L, Nx = 5.0, 50
    dx = L / (Nx - 1)
    dt = 200.0
    steps = min(int((days * 24 * 3600) / dt), 10000)

    x = np.linspace(0, L, Nx)
    T = np.full(Nx, float(T_base))
    T[int(Nx * 0.4):int(Nx * 0.6)] = float(T_hot)

    M = np.full(Nx, float(M_base))
    DML = np.zeros(Nx)
    Fungal = np.zeros(Nx)

    k_wall = 0.8
    h_env = k_wall / max(wall_thickness, 0.05)

    for _ in range(steps):
        d2T = np.zeros(Nx)
        d2M = np.zeros(Nx)

        d2T[1:-1] = (T[2:] - 2*T[1:-1] + T[:-2]) / dx**2
        d2M[1:-1] = (M[2:] - 2*M[1:-1] + M[:-2]) / dx**2

        T_safe = np.clip(T, 0, 85)

        Q_bio = 100.0 * np.exp(0.05 * T_safe) * (M / 14.0)
        dT_dt = alpha * d2T + Q_bio / (rho * Cp)
        dM_dt = Dm * d2M + delta * d2T

        T += dT_dt * dt
        M += dM_dt * dt

        T[0]  += -h_env * (T[0]  - T_base) * dt / (rho * Cp)
        T[-1] += -h_env * (T[-1] - T_base) * dt / (rho * Cp)

        spoilage_rate = (
            spoilage_factor *
            0.00001 * np.exp(0.06 * T_safe) * (M / 12.0)
        )

        DML += spoilage_rate * dt
        Fungal += spoilage_rate * 2.0 * dt

    return x, T, M, DML, Fungal

def spoilage_probability_distribution(max_dml):
    max_dml = float(np.clip(max_dml, 0.0, 1.0))

    alpha = 1.0 + max_dml * 20
    beta = 1.0 + (1 - max_dml) * 20

    dist = tfd.Beta(alpha, beta)
    samples = dist.sample(2000).numpy()

    mean_risk = np.mean(samples)
    low_ci, high_ci = np.percentile(samples, [5, 95])

    return samples, mean_risk, low_ci, high_ci

class GrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Grain Spoilage Prediction | Google AI")

        controls = tk.Frame(root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        tk.Label(controls, text="Crop Type:").pack(side=tk.LEFT)
        self.crop = tk.StringVar(value="Wheat")
        tk.OptionMenu(controls, self.crop, *CROPS.keys()).pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Base Temp (°C):").pack(side=tk.LEFT)
        self.t_base = tk.Entry(controls, width=5)
        self.t_base.insert(0, "25")
        self.t_base.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Hotspot Temp (°C):").pack(side=tk.LEFT)
        self.t_hot = tk.Entry(controls, width=5)
        self.t_hot.insert(0, "40")
        self.t_hot.pack(side=tk.LEFT, padx=5)

        tk.Label(controls, text="Silo Wall Thickness (m):").pack(side=tk.LEFT)
        self.wall = tk.Entry(controls, width=5)
        self.wall.insert(0, "0.3")
        self.wall.pack(side=tk.LEFT, padx=5)

        tk.Button(
            controls,
            text="Run Simulation",
            command=self.run,
            background="#1976D2",
            foreground="white",
            activebackground="#0D47A1",
            activeforeground="white",
            highlightthickness=0,
            bd=0,
            padx=12,
            pady=6
        ).pack(side=tk.LEFT, padx=10)

        self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(
            1, 3, figsize=(12, 4)
        )

        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def run(self):
        try:
            x, T, M, DML, _ = run_simulation(
                days=10,
                T_base=float(self.t_base.get()),
                T_hot=float(self.t_hot.get()),
                M_base=14.0,
                crop=self.crop.get(),
                wall_thickness=float(self.wall.get())
            )

            max_dml = np.max(DML)
            samples, mean_risk, low_ci, high_ci = spoilage_probability_distribution(max_dml)

            for ax in (self.ax1, self.ax2, self.ax3):
                ax.clear()

            self.ax1.plot(x, T, color="red")
            self.ax1.set_title("Temperature Profile (°C)")

            self.ax2.plot(x, M, color="blue")
            self.ax2.set_title("Moisture Profile (%)")

            self.ax3.hist(samples, bins=30, density=True, color="purple", alpha=0.7)
            self.ax3.axvline(mean_risk, color="black", linestyle="--", label="Mean")
            self.ax3.axvline(low_ci, color="red", linestyle=":", label="5% CI")
            self.ax3.axvline(high_ci, color="red", linestyle=":", label="95% CI")
            self.ax3.set_title("Spoilage Risk Distribution")
            self.ax3.legend(fontsize=8, loc="upper left")

            self.fig.subplots_adjust(left=0.06, right=0.98, wspace=0.3)
            self.canvas.draw()

            with open("grain_report.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Height", "Temp", "Moisture", "DML"])
                writer.writerows(np.column_stack((x, T, M, DML)))

            status = "✅ LOW RISK"
            if mean_risk > 0.6:
                status = "🚨 HIGH RISK"
            elif mean_risk > 0.3:
                status = "⚠️ MEDIUM RISK"

            messagebox.showinfo(
                "AI Analysis (Google TensorFlow Probability)",
                f"Status: {status}\n\n"
                f"Mean Risk: {mean_risk:.2f}\n"
                f"95% CI: [{low_ci:.2f}, {high_ci:.2f}]\n\n"
                f"Report saved as grain_report.csv"
            )

        except Exception as e:
            messagebox.showerror("Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = GrainApp(root)
    root.mainloop()
