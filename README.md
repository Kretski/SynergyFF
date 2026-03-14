# SynergyFF: Environment-Aware Fault-Tolerant Force Field Ensemble 🧪💻

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

**SynergyFF** is a dynamic Bayesian ensemble designed to bridge the gap between classical Force Fields (MMFF, UFF) and modern Neural Network Potentials (ANI-2x). 

The core philosophy of this project is **reliability over raw accuracy**. While Machine Learning potentials like ANI-2x offer DFT-level accuracy, they are notoriously brittle—encountering out-of-distribution conformations can lead to massive energy hallucinations. SynergyFF solves this by cross-evaluating multiple models in real-time and dynamically adjusting their trust weights using an **Environment-Aware Kalman Filter**.

## 🚀 Key Features

* **Self-Supervised Disagreement Detection:** The ensemble evaluates the consensus between models on the fly. If one model deviates wildly from the others (e.g., ANI-2x experiencing a domain shift), its trust weight is automatically penalized—*no QM reference required*.
* **Environment-Aware Learning:** The Kalman filter doesn't maintain just one global weight. It dynamically branches trust matrices based on the molecular heavy-atom signature (e.g., "C", "CO", "CN"). The ensemble learns which model is best suited for specific chemical environments.
* **Robust to Extreme Errors:** Capable of maintaining chemical accuracy even when individual underlying models fail catastrophically.

## 📊 Benchmark Results

SynergyFF was rigorously benchmarked against high-level DFT references (B3LYP/6-31G* and ωB97M-D3BJ) across multiple datasets:

| Benchmark Test | Best Single Model Error (MAE) | SynergyFF Error (MAE) | Improvement |
| :--- | :--- | :--- | :--- |
| **SPICE Dataset (100 mols)** | UFF (0.088 kcal/mol) *[ANI-2x failed: 91.5 kcal/mol]* | **0.273 kcal/mol** | Successfully bypassed severe ML hallucinations |
| **ORCA Test 1 (Small)** | MMFF/ANI (0.002 kcal/mol) | **0.036 kcal/mol** | Maintained high accuracy |
| **Torsion Barriers (Real ΔE)** | ANI-2x (5.433 kcal/mol) | **3.073 kcal/mol** | **+43%** better than the best individual model |

*(Note: In Torsion barrier benchmarking—the true test of conformational energetics—the ensemble significantly outperformed every individual method).*

## ⚙️ Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/Kretski/SynergyFF.git](https://github.com/yourusername/SynergyFF.git)
   cd SynergyFF
Install dependencies (it is highly recommended to use a virtual environment):

Bash
pip install -r requirements.txt
Note: Due to RDKit compatibility, numpy<2 is enforced.

🛠️ Usage
The project includes a full benchmarking suite that can automatically download the SPICE dataset and generate ORCA inputs.

To run the complete evaluation pipeline:

Bash
# 1. Generate ORCA inputs and download SPICE data
python synergy_ff.py --generate

# 2. Run Torsion and SPICE benchmarks
python synergy_ff.py --torsion
python synergy_ff.py --spice

# 3. (Optional) Run ORCA benchmark if you have calculated the .out files
python synergy_ff.py --benchmark

# 4. Generate the final combined report and plots
python synergy_ff.py --summary
Production API Example
You can easily integrate the SynergyFF class into your own pipelines for zero-shot energy prediction without any QM reference:

Python
from synergy_ff import SynergyFF, load_ani2x

ani_model, converter = load_ani2x()
ff = SynergyFF(ani_model, converter)

# Predict relative energies. Trust weights are updated automatically under the hood!
result = ff.predict("CC(C)Cc1ccc(cc1)C(C)C(=O)O", conf_id=0)

print(f"Ensemble Energy: {result['ensemble']} kcal/mol")
print(f"Most Trusted Model for this environment: {result['most_trusted']}")
🧠 Future Roadmap
Phase 1: Implementation of Geometry Optimization (calculating ensemble forces/gradients for L-BFGS).

Phase 2: Protein-Ligand Complex evaluation (Docking Rescoring).

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.

⭐ 2   [Stars badge]
