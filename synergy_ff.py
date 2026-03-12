"""
SynergyFF — Environment-Aware Fault-Tolerant Force Field Ensemble
=====================================================================
A dynamic Bayesian ensemble bridging classical Force Fields (MMFF, UFF) 
and Neural Network Potentials (ANI-2x).

Features:
  - Self-supervised disagreement detection (no QM reference needed)
  - Chemical environment-aware Kalman filtering
  - Torsion barrier benchmarking
  - SPICE & ORCA dataset evaluation
"""

import os
import sys
import math
import warnings
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

# ═══════════════════════════════════════════════════════
#  КОНСТАНТИ
# ═══════════════════════════════════════════════════════

HARTREE_TO_KCAL = 627.5094740631
ANI2X_SUPPORTED = {1, 6, 7, 8, 9, 16, 17}
ANI2X_SYMBOLS   = {1:'H', 6:'C', 7:'N', 8:'O', 9:'F', 16:'S', 17:'Cl'}

SPICE_URL  = "https://zenodo.org/record/8222043/files/SPICE-1.1.4.hdf5"
SPICE_FILE = "./spice_data/SPICE-1.1.4.hdf5"

# ═══════════════════════════════════════════════════════
#  МОЛЕКУЛИ ЗА ORCA ТЕСТ 1 и 2
# ═══════════════════════════════════════════════════════

ORCA_TEST1 = [
    ("methane",       "C"),
    ("ethane",        "CC"),
    ("propane",       "CCC"),
    ("methanol",      "CO"),
    ("ethanol",       "CCO"),
    ("acetaldehyde",  "CC=O"),
    ("acetone",       "CC(C)=O"),
    ("acetic_acid",   "CC(=O)O"),
    ("methylamine",   "CN"),
    ("dimethylether", "COC"),
]

ORCA_TEST2 = [
    ("ibuprofen",   "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("aspirin",     "CC(=O)Oc1ccccc1C(=O)O"),
    ("paracetamol", "CC(=O)Nc1ccc(O)cc1"),
    ("salicylic",   "OC(=O)c1ccccc1O"),
    ("caffeine",    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"),
]

# ═══════════════════════════════════════════════════════
#  ТОРСИОННИ МОЛЕКУЛИ — Test3
# ═══════════════════════════════════════════════════════

TORSION_MOLECULES = [
    {
        "name":   "ethane",
        "smiles": "CC",
        "torsion_smarts": "[H][C][C][H]",  
        "n_points": 12,   
        "ref_barrier_kcal": 2.9,   
    },
    {
        "name":   "butane",
        "smiles": "CCCC",
        "torsion_smarts": "[C][C][C][C]",
        "n_points": 12,
        "ref_barrier_kcal": 4.5,   
    },
    {
        "name":   "propanol",
        "smiles": "CCCO",
        "torsion_smarts": "[C][C][C][O]",
        "n_points": 12,
        "ref_barrier_kcal": 3.3,
    },
    {
        "name":   "dimethylether",
        "smiles": "COC",
        "torsion_smarts": "[H][C][O][C]",
        "n_points": 12,
        "ref_barrier_kcal": 2.7,
    },
    {
        "name":   "methylformate",
        "smiles": "COC=O",
        "torsion_smarts": "[C][O][C]=[O]",
        "n_points": 12,
        "ref_barrier_kcal": 4.8,
    },
    {
        "name":   "acetaldehyde",
        "smiles": "CC=O",
        "torsion_smarts": "[H][C][C]=[O]",
        "n_points": 12,
        "ref_barrier_kcal": 1.2,
    },
]

# ═══════════════════════════════════════════════════════
#  ANI-2x
# ═══════════════════════════════════════════════════════

def load_ani2x():
    try:
        import torchani
        print("  ANI-2x: Loading...")
        model = torchani.models.ANI2x(periodic_table_index=False)
        model.eval()
        converter = torchani.utils.ChemicalSymbolsToInts(
            ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']
        )
        print("  ANI-2x: OK ✓")
        return model, converter
    except Exception as e:
        print(f"  ANI-2x Error: {e}")
        return None, None

def check_ani2x(mol):
    return all(a.GetAtomicNum() in ANI2X_SUPPORTED for a in mol.GetAtoms())

def get_ani2x_energy(mol, ani_model, converter, conf_id=0):
    if ani_model is None or not check_ani2x(mol):
        return None
    try:
        conf    = mol.GetConformer(conf_id)
        pos     = torch.tensor(conf.GetPositions(), dtype=torch.float32).unsqueeze(0)
        syms    = [ANI2X_SYMBOLS[a.GetAtomicNum()] for a in mol.GetAtoms()]
        species = converter(syms).unsqueeze(0)
        with torch.no_grad():
            e_h = ani_model((species, pos)).energies.item()
        return e_h * HARTREE_TO_KCAL
    except Exception:
        return None

# ═══════════════════════════════════════════════════════
#  FORCE FIELDS
# ═══════════════════════════════════════════════════════

def get_mmff_energy(mol, conf_id=0):
    try:
        mol.UpdatePropertyCache(strict=False)
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            pass
        props = AllChem.MMFFGetMoleculeProperties(mol)
        if props is None:
            return None
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        return ff.CalcEnergy() if ff else None
    except Exception:
        return None

def get_uff_energy(mol, conf_id=0):
    try:
        ff = AllChem.UFFGetMoleculeForceField(mol, conf_id)
        return ff.CalcEnergy() if ff else None
    except Exception:
        return None

def generate_conformations(smiles, n_confs=2, seed=42):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, []
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.numThreads = 1
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        return None, []
    for cid in conf_ids:
        try:
            AllChem.MMFFOptimizeMolecule(mol, confId=cid)
        except Exception:
            pass
    return mol, list(conf_ids)

# ═══════════════════════════════════════════════════════
#  TORSION SCAN
# ═══════════════════════════════════════════════════════

def find_torsion_atoms(mol, smarts):
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None:
        return None
    matches = mol.GetSubstructMatches(pattern)
    if not matches:
        return None
    return matches[0][:4]

def scan_torsion(smiles, torsion_smarts, n_points=12, ani_model=None, converter=None):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None: return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    AllChem.EmbedMolecule(mol, params)

    torsion_atoms = find_torsion_atoms(mol, torsion_smarts)
    if torsion_atoms is None:
        ri = mol.GetRingInfo()
        for bond in mol.GetBonds():
            if bond.GetBondTypeAsDouble() == 1.0:
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if not ri.NumAtomRings(i) and not ri.NumAtomRings(j):
                    ni = [a.GetIdx() for a in mol.GetAtomWithIdx(i).GetNeighbors() if a.GetIdx() != j]
                    nj = [a.GetIdx() for a in mol.GetAtomWithIdx(j).GetNeighbors() if a.GetIdx() != i]
                    if ni and nj:
                        torsion_atoms = (ni[0], i, j, nj[0])
                        break
    if torsion_atoms is None: return None

    a1, a2, a3, a4 = torsion_atoms
    angles = np.linspace(0, 360, n_points, endpoint=False)
    rows   = []

    for angle in angles:
        from copy import deepcopy
        mol_copy = deepcopy(mol)

        try:
            rdMolTransforms.SetDihedralDeg(mol_copy.GetConformer(), a1, a2, a3, a4, angle)
        except Exception:
            continue

        e_mmff = get_mmff_energy(mol_copy, 0)
        e_uff  = get_uff_energy(mol_copy, 0)
        e_ani  = get_ani2x_energy(mol_copy, ani_model, converter, 0)

        rows.append({
            "angle":   round(angle, 1),
            "e_mmff":  e_mmff,
            "e_uff":   e_uff,
            "e_ani":   e_ani,
        })

    if not rows: return None

    df = pd.DataFrame(rows)
    for col in ["e_mmff", "e_uff", "e_ani"]:
        if df[col].notna().any():
            df[col] = df[col] - df[col].dropna().min()

    return df

def run_torsion_benchmark(ani_model, converter):
    print(f"\n{'='*70}")
    print("TORSION BENCHMARK (Test3) — Real ΔE 1-15 kcal/mol")
    print(f"{'='*70}")

    kalman  = KalmanTrust(["MMFF", "UFF", "ANI"])
    results = []
    scan_data = {}

    for mol_info in TORSION_MOLECULES:
        name   = mol_info["name"]
        smiles = mol_info["smiles"]
        smarts = mol_info["torsion_smarts"]
        n_pts  = mol_info["n_points"]
        ref_barrier = mol_info["ref_barrier_kcal"]

        df = scan_torsion(smiles, smarts, n_pts, ani_model, converter)
        if df is None or df.empty:
            continue

        scan_data[name] = df
        mol_obj = Chem.MolFromSmiles(smiles)

        b_mmff = df["e_mmff"].max() if df["e_mmff"].notna().any() else None
        b_uff  = df["e_uff"].max()  if df["e_uff"].notna().any()  else None
        b_ani  = df["e_ani"].max()  if df["e_ani"].notna().any()  else None

        for name_k, b in [("MMFF", b_mmff), ("UFF", b_uff), ("ANI", b_ani)]:
            if b is not None:
                kalman.update(name_k, abs(b - ref_barrier), mol_obj)

        weights = kalman.weights(mol_obj)
        ens_parts, wsum = 0.0, 0.0
        for name_k, b in [("MMFF", b_mmff), ("UFF", b_uff), ("ANI", b_ani)]:
            if b is not None:
                ens_parts += weights[name_k] * b
                wsum      += weights[name_k]
        b_ens = ens_parts / (wsum + 1e-8)

        err_mmff = abs(b_mmff - ref_barrier) if b_mmff is not None else None
        err_uff  = abs(b_uff  - ref_barrier) if b_uff  is not None else None
        err_ani  = abs(b_ani  - ref_barrier) if b_ani  is not None else None
        err_ens  = abs(b_ens  - ref_barrier)

        print(f"  {name:<15} | Ref: {ref_barrier:.1f} | "
              f"MMFF: {b_mmff:.2f} | ANI: {b_ani:.2f} | ENS: {b_ens:.2f}")

        results.append({
            "molecule":   name,
            "ref":        ref_barrier,
            "b_mmff":     b_mmff,
            "b_uff":      b_uff,
            "b_ani":      b_ani,
            "b_ens":      b_ens,
            "err_mmff":   err_mmff,
            "err_uff":    err_uff,
            "err_ani":    err_ani,
            "err_ens":    err_ens,
            "w_mmff":     weights["MMFF"],
            "w_uff":      weights["UFF"],
            "w_ani":      weights["ANI"],
        })

    return pd.DataFrame(results), kalman, scan_data

# ═══════════════════════════════════════════════════════
#  KALMAN TRUST (Environment-Aware)
# ═══════════════════════════════════════════════════════

class KalmanTrust:
    """
    Environment-aware Bayesian trust estimator via Kalman filter.
    Branches trust matrices based on heavy-atom environments (e.g., 'C', 'CO', 'CN').
    """
    def __init__(self, models, R=0.05, Q=0.005):
        self.models = models
        self.R = R
        self.Q = Q
        self.trust   = {}
        self.P       = {}
        self.history = {}

    def _get_env(self, mol):
        if mol is None:
            return "Global"
        syms = set(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() > 1)
        return "".join(sorted(list(syms))) if syms else "Global"

    def _init_env(self, env):
        if env not in self.trust:
            self.trust[env]   = {m: 1.0 for m in self.models}
            self.P[env]       = {m: 1.0 for m in self.models}
            self.history[env] = {m: [] for m in self.models}

    def update(self, name, error, mol=None):
        env = self._get_env(mol)
        self._init_env(env)

        K = self.P[env][name] / (self.P[env][name] + self.R)
        self.trust[env][name] += K * (1.0 / (error + 1e-6) - self.trust[env][name])
        self.P[env][name]      = (1 - K) * self.P[env][name] + self.Q

        for m in self.models:
            self.history[env][m].append(self.trust[env][m])

    def update_self_supervised(self, predictions: dict, mol=None):
        valid = {k: v for k, v in predictions.items() if v is not None}
        if len(valid) < 2: return

        w = self.weights(mol)
        ens = sum(w[k] * v for k, v in valid.items()) / (sum(w[k] for k in valid) + 1e-8)

        for name, val in valid.items():
            self.update(name, abs(val - ens), mol)

    def weights(self, mol=None):
        env = self._get_env(mol)
        self._init_env(env)
        total = sum(self.trust[env].values())
        return {k: v / total for k, v in self.trust[env].items()}

    def most_trusted(self, mol=None) -> str:
        env = self._get_env(mol)
        self._init_env(env)
        return max(self.trust[env], key=self.trust[env].get)

# ═══════════════════════════════════════════════════════
#  PRODUCTION API
# ═══════════════════════════════════════════════════════

class SynergyFF:
    def __init__(self, ani_model, converter, R=0.05, Q=0.005):
        self.ani_model = ani_model
        self.converter = converter
        self.kalman    = KalmanTrust(["MMFF", "UFF", "ANI"], R=R, Q=Q)
        self.n_calls   = 0

    def _mol_from_smiles(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        AllChem.EmbedMolecule(mol, params)
        try: AllChem.MMFFOptimizeMolecule(mol)
        except Exception: pass
        return mol

    def predict(self, smiles, conf_id=0) -> dict:
        mol = self._mol_from_smiles(smiles) if isinstance(smiles, str) else smiles
        if mol is None: return None

        e_mmff = get_mmff_energy(mol, conf_id)
        e_uff  = get_uff_energy(mol, conf_id)
        e_ani  = get_ani2x_energy(mol, self.ani_model, self.converter, conf_id)

        self.kalman.update_self_supervised(
            {"MMFF": e_mmff, "UFF": e_uff, "ANI": e_ani}, mol
        )
        self.n_calls += 1

        w     = self.kalman.weights(mol)
        valid = {k: v for k, v in [("MMFF", e_mmff), ("UFF", e_uff), ("ANI", e_ani)] if v is not None}
        ens   = sum(w[k] * v for k, v in valid.items()) / (sum(w[k] for k in valid) + 1e-8)

        return {
            "ensemble":     round(ens, 4),
            "mmff":         round(e_mmff, 4) if e_mmff is not None else None,
            "uff":          round(e_uff,  4) if e_uff  is not None else None,
            "ani":          round(e_ani,  4) if e_ani  is not None else None,
            "w_mmff":       round(w["MMFF"], 4),
            "w_uff":        round(w["UFF"],  4),
            "w_ani":        round(w["ANI"],  4),
            "most_trusted": self.kalman.most_trusted(mol),
            "env":          self.kalman._get_env(mol),
        }

# ═══════════════════════════════════════════════════════
#  ORCA GENERATOR
# ═══════════════════════════════════════════════════════

ORCA_TEMPLATE = """! B3LYP 6-31G* TightSCF RIJCOSX def2/J
%maxcore 4000

* xyz {charge} {mult}
{xyz}
*

"""

def mol_to_xyz_block(mol, conf_id):
    conf = mol.GetConformer(conf_id)
    lines = []
    for atom in mol.GetAtoms():
        p = conf.GetAtomPosition(atom.GetIdx())
        lines.append(f"  {atom.GetSymbol():<3} {p.x:12.6f} {p.y:12.6f} {p.z:12.6f}")
    return "\n".join(lines)

def write_orca_inputs(molecules, output_dir, n_confs=2, test_name="test"):
    os.makedirs(output_dir, exist_ok=True)
    jobs = []
    for mol_name, smiles in molecules:
        mol, conf_ids = generate_conformations(smiles, n_confs=n_confs)
        if mol is None or len(conf_ids) < 2: continue
        if not check_ani2x(mol): continue
        for cid in conf_ids:
            job_name = f"{mol_name}_conf{cid}"
            inp_path = os.path.join(output_dir, f"{job_name}.inp")
            with open(inp_path, "w") as f:
                f.write(ORCA_TEMPLATE.format(
                    charge=0, mult=1, xyz=mol_to_xyz_block(mol, cid)
                ))
            jobs.append({"job_name": job_name})

    bat = os.path.join(output_dir, f"run_{test_name}.bat")
    with open(bat, "w") as f:
        f.write("@echo off\ncd /d %~dp0\n")
        for job in jobs:
            f.write(f'"C:\\ORCA\\orca.exe" {job["job_name"]}.inp\n')
        f.write("echo Ready!\npause\n")
    print(f"  ORCA jobs prepared: {len(jobs)} in {output_dir}")

def parse_orca_energy(out_path):
    if os.path.exists(out_path):
        try:
            with open(out_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "FINAL SINGLE POINT ENERGY" in line:
                        return float(line.strip().split()[-1])
        except Exception: pass
    return None

def orca_completed(out_path):
    if os.path.exists(out_path):
        try:
            if "ORCA TERMINATED NORMALLY" in open(out_path, encoding="utf-8", errors="ignore").read():
                return True
        except Exception: pass
    return False

# ═══════════════════════════════════════════════════════
#  ENSEMBLE CALCULATOR
# ═══════════════════════════════════════════════════════

def calc_ensemble_delta_e(mol_a, mol_b, cid_a, cid_b, ani_model, converter, kalman, qm_de):
    mm_a    = get_mmff_energy(mol_a, cid_a)
    mm_b    = get_mmff_energy(mol_b, cid_b)
    mmff_de = (mm_b - mm_a) if (mm_a is not None and mm_b is not None) else None

    uf_a   = get_uff_energy(mol_a, cid_a)
    uf_b   = get_uff_energy(mol_b, cid_b)
    uff_de = (uf_b - uf_a) if (uf_a is not None and uf_b is not None) else None

    an_a   = get_ani2x_energy(mol_a, ani_model, converter, cid_a)
    an_b   = get_ani2x_energy(mol_b, ani_model, converter, cid_b)
    ani_de = (an_b - an_a) if (an_a is not None and an_b is not None) else None

    if ani_de is None: return None

    for name, de in [("MMFF", mmff_de), ("UFF", uff_de), ("ANI", ani_de)]:
        if de is not None:
            kalman.update(name, abs(de - qm_de), mol_a)

    weights = kalman.weights(mol_a)
    total, wsum = 0.0, 0.0
    for name, de in [("MMFF", mmff_de), ("UFF", uff_de), ("ANI", ani_de)]:
        if de is not None:
            total += weights[name] * de
            wsum  += weights[name]
    ens_de = total / (wsum + 1e-8)

    return {
        "qm_de":    round(qm_de,   4),
        "mmff_de":  round(mmff_de, 4) if mmff_de is not None else None,
        "uff_de":   round(uff_de,  4) if uff_de  is not None else None,
        "ani_de":   round(ani_de,  4),
        "ens_de":   round(ens_de,  4),
        "err_mmff": round(abs(mmff_de - qm_de), 4) if mmff_de is not None else None,
        "err_uff":  round(abs(uff_de  - qm_de), 4) if uff_de  is not None else None,
        "err_ani":  round(abs(ani_de  - qm_de), 4),
        "err_ens":  round(abs(ens_de  - qm_de), 4),
        "w_mmff":   round(weights["MMFF"], 4),
        "w_uff":    round(weights["UFF"],  4),
        "w_ani":    round(weights["ANI"],  4),
    }

# ═══════════════════════════════════════════════════════
#  ORCA BENCHMARK (Test1 + Test2)
# ═══════════════════════════════════════════════════════

def run_orca_benchmark(molecules, output_dir, ani_model, converter, test_name):
    print(f"\n{'='*70}")
    print(f"ORCA BENCHMARK — B3LYP/6-31G* | {test_name}")
    print(f"{'='*70}")

    kalman  = KalmanTrust(["MMFF", "UFF", "ANI"])
    results = []

    for mol_name, smiles in molecules:
        out_a = os.path.join(output_dir, f"{mol_name}_conf0.out")
        out_b = os.path.join(output_dir, f"{mol_name}_conf1.out")

        if not orca_completed(out_a) or not orca_completed(out_b): continue

        e_a, e_b = parse_orca_energy(out_a), parse_orca_energy(out_b)
        if e_a is None or e_b is None: continue

        qm_de = (e_b - e_a) * HARTREE_TO_KCAL
        mol, conf_ids = generate_conformations(smiles, n_confs=2)
        if mol is None or len(conf_ids) < 2: continue

        res = calc_ensemble_delta_e(mol, mol, conf_ids[0], conf_ids[1], ani_model, converter, kalman, qm_de)
        if res is None: continue

        res["molecule"] = mol_name
        results.append(res)
        w = kalman.weights(mol)
        print(f"  {mol_name:<20} | QM: {qm_de:7.3f} | ENS err: {res['err_ens']:6.3f}")

    return pd.DataFrame(results), kalman

# ═══════════════════════════════════════════════════════
#  SPICE DATASET
# ═══════════════════════════════════════════════════════

def download_spice():
    os.makedirs("./spice_data", exist_ok=True)
    if os.path.exists(SPICE_FILE): return True
    try:
        import requests
        print("  Downloading SPICE dataset...")
        r = requests.get(SPICE_URL, stream=True, timeout=30)
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(SPICE_FILE, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                done += len(chunk)
                if total > 0: print(f"\r  {done/total*100:.1f}%", end="", flush=True)
        print("\n  SPICE: OK")
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False

def load_spice_molecules(max_molecules=100, max_atoms=30):
    try: import h5py
    except ImportError: return []

    if not os.path.exists(SPICE_FILE): return []
    results = []

    try:
        with h5py.File(SPICE_FILE, "r") as f:
            mol_names = list(f.keys())
            for mol_name in mol_names:
                if len(results) >= max_molecules: break
                try:
                    grp       = f[mol_name]
                    atom_nums = grp["atomic_numbers"][:]
                    confs     = grp["conformations"][:]
                    energies  = grp["formation_energy"][:]

                    if not all(int(z) in ANI2X_SUPPORTED for z in atom_nums): continue
                    if len(atom_nums) < 5 or len(atom_nums) > max_atoms: continue
                    if len(confs) < 2: continue

                    e_kcal = energies * HARTREE_TO_KCAL
                    delta_e_matrix = np.abs(e_kcal[:, None] - e_kcal[None, :])
                    np.fill_diagonal(delta_e_matrix, 999.0)
                    i, j = np.unravel_index(delta_e_matrix.argmin(), delta_e_matrix.shape)

                    qm_de = float(e_kcal[j] - e_kcal[i])
                    if abs(qm_de) < 0.05 or abs(qm_de) > 5.0: continue

                    results.append({
                        "name":      mol_name,
                        "atom_nums": atom_nums.tolist(),
                        "conf_a":    confs[i].tolist(),
                        "conf_b":    confs[j].tolist(),
                        "e_a":       float(e_kcal[i]),
                        "e_b":       float(e_kcal[j]),
                        "qm_de":     qm_de,
                    })
                except Exception: continue
    except Exception as e:
        return []

    return results

def spice_to_rdkit_mol(atom_nums, coords):
    try:
        from rdkit.Chem import RWMol, Atom
        from rdkit.Geometry import Point3D
        from rdkit.Chem import rdDetermineBonds

        rw = RWMol()
        for z in atom_nums: rw.AddAtom(Atom(int(z)))
        conf = Chem.Conformer(len(atom_nums))
        for i, (x, y, z_) in enumerate(coords):
            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z_)))
        rw.AddConformer(conf, assignId=True)
        try:
            rdDetermineBonds.DetermineConnectivity(rw)
            rdDetermineBonds.DetermineBondOrders(rw, charge=0)
        except Exception:
            try: rdDetermineBonds.DetermineConnectivity(rw)
            except Exception: pass
        mol = rw.GetMol()
        try: Chem.SanitizeMol(mol)
        except Exception: pass
        mol.UpdatePropertyCache(strict=False)
        return mol
    except Exception: return None

def run_spice_benchmark(ani_model, converter, max_molecules=100):
    print(f"\n{'='*70}")
    print("SPICE BENCHMARK — ωB97M-D3BJ/def2-TZVPPD reference")
    print(f"{'='*70}")

    spice_mols = load_spice_molecules(max_molecules=max_molecules)
    if not spice_mols: return pd.DataFrame(), KalmanTrust(["MMFF", "UFF", "ANI"])

    kalman  = KalmanTrust(["MMFF", "UFF", "ANI"])
    results = []

    for idx, data in enumerate(spice_mols):
        mol_a = spice_to_rdkit_mol(data["atom_nums"], data["conf_a"])
        mol_b = spice_to_rdkit_mol(data["atom_nums"], data["conf_b"])
        if mol_a is None or mol_b is None: continue

        res = calc_ensemble_delta_e(mol_a, mol_b, 0, 0, ani_model, converter, kalman, data["qm_de"])
        if res is None: continue

        res["molecule"] = data["name"]
        results.append(res)

        if (idx + 1) % 10 == 0:
            print(f"  {idx+1:3d}/{len(spice_mols)} | QM:{data['qm_de']:6.3f} | ENS err:{res['err_ens']:6.3f}")

    return pd.DataFrame(results), kalman

# ═══════════════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ — ORCA/SPICE
# ═══════════════════════════════════════════════════════

def plot_benchmark(df, title, output_path):
    if df.empty: return {}
    mae_raw = {
        "MMFF":     df["err_mmff"].dropna().mean(),
        "UFF":      df["err_uff"].dropna().mean(),
        "ANI-2x":   df["err_ani"].mean(),
        "Ensemble": df["err_ens"].mean(),
    }
    mae = {k: v for k, v in mae_raw.items() if pd.notna(v) and np.isfinite(v)}
    if not mae: return mae_raw

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    colors_map = {"MMFF":"#4472C4","UFF":"#FF0000","ANI-2x":"#70AD47","Ensemble":"#FFC000"}

    ax = axes[0]
    lbls, vals = list(mae.keys()), list(mae.values())
    clrs = [colors_map.get(l, "#888") for l in lbls]
    bars = ax.bar(lbls, vals, color=clrs, alpha=0.85, edgecolor="black", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                f"{val:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"MAE vs QM\n{title} (n={len(df)})")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_ylim(0, max(vals) * 1.4 + 0.001)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ax.scatter(df["qm_de"], df["ani_de"], color="green", alpha=0.6, s=50, label=f"ANI-2x (MAE={mae.get('ANI-2x',0):.3f})")
    ax.scatter(df["qm_de"], df["ens_de"], color="orange", alpha=0.7, s=60, marker="^", label=f"Ensemble (MAE={mae.get('Ensemble',0):.3f})")
    lim = max(df["qm_de"].abs().max(), df["ani_de"].abs().max(), df["ens_de"].abs().max()) * 1.2 + 0.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="Perfect")
    ax.set_xlabel("QM ΔE (kcal/mol)")
    ax.set_ylabel("Predicted ΔE (kcal/mol)")
    ax.set_title(f"Predicted vs QM\n{title}")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return mae_raw

def plot_qm_vs_synergy(df, output_path):
    if df.empty or "qm_de" not in df or "ens_de" not in df: return

    qm   = df["qm_de"].dropna().values
    pred = df["ens_de"].dropna().values

    if len(qm) == 0 or len(pred) == 0: return

    plt.figure(figsize=(6,6))
    plt.scatter(qm, pred, alpha=0.7, color="#FFC000", edgecolor="black", s=50)

    lim_min = min(qm.min(), pred.min()) - 0.5
    lim_max = max(qm.max(), pred.max()) + 0.5
    plt.plot([lim_min, lim_max], [lim_min, lim_max], "k--", lw=1.5, label="Perfect Match")

    plt.xlabel("QM ΔE (kcal/mol)", fontsize=11)
    plt.ylabel("SynergyFF prediction (kcal/mol)", fontsize=11)
    
    rmse = np.sqrt(np.mean((qm - pred)**2))
    variance = np.var(qm)
    
    if variance < 0.01:
        plt.title(f"QM vs SynergyFF Correlation\nRMSE = {rmse:.3f} kcal/mol", fontsize=12)
    else:
        try: r2 = r2_score(qm, pred)
        except: r2 = float("nan")
        plt.title(f"QM vs SynergyFF Correlation\n$R^2$ = {r2:.3f} | RMSE = {rmse:.3f} kcal/mol", fontsize=12)

    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_qm_vs_predicted(all_results, output_path):
    qm_all, ens_all, ani_all = [], [], []

    for test_name, (df, _) in all_results.items():
        if "ORCA" not in test_name.replace("\n", " "): continue
        if "qm_de" not in df.columns: continue
        qm_all.extend(df["qm_de"].tolist())
        ens_all.extend(df["ens_de"].tolist())
        ani_all.extend(df["ani_de"].tolist())

    if len(qm_all) < 3: return

    qm, ens, ani = np.array(qm_all), np.array(ens_all), np.array(ani_all)

    def metrics(pred):
        mae  = np.mean(np.abs(pred - qm))
        rmse = np.sqrt(np.mean((pred - qm)**2))
        if np.var(qm) < 0.01: r2 = float("nan")
        else:
            try: r2 = r2_score(qm, pred)
            except: r2 = float("nan")
        return mae, rmse, r2

    mae_e, rmse_e, r2_e = metrics(ens)
    mae_a, rmse_a, r2_a = metrics(ani)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    lim = max(np.abs(qm).max(), np.abs(ens).max(), np.abs(ani).max()) * 1.2 + 0.05

    for ax, pred, color, label, mae, rmse, r2 in [
        (axes[0], ens, "#FFC000", "Ensemble", mae_e, rmse_e, r2_e),
        (axes[1], ani, "#70AD47", "ANI-2x",   mae_a, rmse_a, r2_a),
    ]:
        ax.scatter(qm, pred, color=color, alpha=0.75, s=80, edgecolors="black", lw=0.5)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, label="Perfect (y=x)")
        ax.set_xlabel("QM ΔE B3LYP/6-31G* (kcal/mol)", fontsize=11)
        ax.set_ylabel(f"{label} ΔE (kcal/mol)", fontsize=11)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        title_str = f"{label} vs QM\nMAE={mae:.4f} | RMSE={rmse:.4f}"
        if not np.isnan(r2): title_str += f" | R²={r2:.3f}"
        ax.set_title(title_str, fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SynergyFF — QM vs Predicted (ORCA Test1+2)", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_kalman_evolution(kalman, output_path):
    envs = [e for e, hist in kalman.history.items() if any(len(v) > 0 for v in hist.values())]
    n = len(envs)
    if n == 0: return

    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    
    if n == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)
    
    colors_map = {"MMFF":"#4472C4", "UFF":"#FF0000", "ANI":"#70AD47"}

    for idx, env in enumerate(envs):
        ax = axes[idx // cols][idx % cols]
        for name, vals in kalman.history[env].items():
            if vals:
                ax.plot(vals, label=name, color=colors_map.get(name, "#888"), lw=2)
        
        ax.set_xlabel("Iterations")
        ax.set_ylabel("Trust Weight")
        env_label = "Global" if env == "Global" else f"Env: {env}"
        ax.set_title(env_label, fontsize=11)
        ax.legend()
        ax.grid(True, alpha=0.3)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("Kalman Trust Evolution by Chemical Environment", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def plot_kalman_trust_evolution(all_results, output_path):
    best_df = None
    best_name = ""
    for test_name, (df, _) in all_results.items():
        name_clean = test_name.replace("\n", " ")
        if "ORCA" in name_clean and "w_mmff" in df.columns:
            if best_df is None or "drug" in name_clean.lower():
                best_df   = df
                best_name = name_clean

    if best_df is None or "w_mmff" not in best_df.columns: return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    x  = range(len(best_df))
    ax.plot(x, best_df["w_mmff"], "o-", color="#4472C4", lw=2, ms=7, label="MMFF")
    ax.plot(x, best_df["w_uff"],  "s-", color="#FF0000",  lw=2, ms=7, label="UFF")
    ax.plot(x, best_df["w_ani"],  "^-", color="#70AD47",  lw=2, ms=7, label="ANI-2x")
    ax.axhline(1/3, color="gray", lw=1, ls="--", label="Equal weights (1/3)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(best_df["molecule"].tolist(), rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Kalman Trust Weight", fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title(f"Dynamic Trust Adaptation\n{best_name}", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    final = best_df.iloc[-1]
    methods = ["MMFF", "UFF", "ANI-2x"]
    weights = [final["w_mmff"], final["w_uff"], final["w_ani"]]
    colors  = ["#4472C4", "#FF0000", "#70AD47"]
    bars = ax.bar(methods, weights, color=colors, alpha=0.85, edgecolor="black", width=0.5)
    for bar, val in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", fontsize=13, fontweight="bold")
    ax.axhline(1/3, color="gray", lw=1.5, ls="--", label="Equal weights")
    ax.set_ylabel("Final Trust Weight", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Final Kalman Weights\n(after {len(best_df)} molecules)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("SynergyFF — Model Reliability Learned Automatically", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_combined_summary(results_dict, output_path):
    n = len(results_dict)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(7*n, 5))
    if n == 1: axes = [axes]
    colors_map = {"MMFF":"#4472C4","UFF":"#FF0000","ANI-2x":"#70AD47","Ensemble":"#FFC000"}

    for ax, (test_name, (df, mae)) in zip(axes, results_dict.items()):
        if df.empty: continue
        mae_clean = {k: v for k, v in mae.items() if pd.notna(v) and np.isfinite(v)}
        if not mae_clean: continue
        lbls, vals = list(mae_clean.keys()), list(mae_clean.values())
        clrs = [colors_map.get(l, "#888") for l in lbls]
        bars = ax.bar(lbls, vals, color=clrs, alpha=0.85, edgecolor="black", width=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                    f"{val:.3f}", ha="center", fontsize=10, fontweight="bold")
        ax.set_title(f"{test_name.replace(chr(10),' ')}\n(n={len(df)})", fontsize=11)
        ax.set_ylabel("MAE (kcal/mol)")
        ax.set_ylim(0, max(vals) * 1.4 + 0.001)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("SynergyFF — Final Benchmark vs QM", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def compute_full_metrics(all_results):
    print(f"\n{'='*75}")
    print("FULL METRICS — MAE | RMSE | R²")
    print(f"{'='*75}")
    print(f"{'Test':<28} {'Method':<10} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print("-" * 75)

    for test_name, (df, _) in all_results.items():
        name = test_name.replace("\n", " ")
        if "qm_de" not in df.columns and "ref" not in df.columns: continue
        is_torsion = "Torsion" in name

        if is_torsion:
            ref = df["ref"].values
            pairs = [
                ("MMFF",     df["b_mmff"].values),
                ("UFF",      df["b_uff"].values),
                ("ANI-2x",   df["b_ani"].values),
                ("Ensemble", df["b_ens"].values),
            ]
        else:
            ref = df["qm_de"].values
            pairs = [
                ("MMFF",     df["mmff_de"].values if "mmff_de" in df else None),
                ("UFF",      df["uff_de"].values  if "uff_de"  in df else None),
                ("ANI-2x",   df["ani_de"].values),
                ("Ensemble", df["ens_de"].values),
            ]

        for method, pred_raw in pairs:
            if pred_raw is None: continue
            mask = ~np.isnan(pred_raw.astype(float))
            if mask.sum() < 2: continue
            pred = pred_raw[mask].astype(float)
            ref_m = ref[mask]

            mae  = np.mean(np.abs(pred - ref_m))
            rmse = np.sqrt(np.mean((pred - ref_m)**2))
            
            if np.var(ref_m) < 0.01: r2 = float("nan")
            else:
                try: r2 = r2_score(ref_m, pred)
                except: r2 = float("nan")

            marker = " ✅" if method == "Ensemble" else ""
            r2_str = f"{r2:8.3f}" if not np.isnan(r2) else "     N/A"
            print(f"  {name:<26} {method:<10} {mae:8.4f} {rmse:8.4f} {r2_str}{marker}")
    print(f"{'='*75}")

def plot_torsion_profiles(scan_data, output_dir="./results"):
    os.makedirs(output_dir, exist_ok=True)
    n = len(scan_data)
    if n == 0: return

    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
    if n == 1: axes = np.array([[axes]])
    elif rows == 1: axes = axes.reshape(1, -1)

    colors = {"e_mmff": "#4472C4", "e_uff": "#FF0000", "e_ani": "#70AD47"}
    labels = {"e_mmff": "MMFF94", "e_uff": "UFF", "e_ani": "ANI-2x"}

    for idx, (mol_name, df) in enumerate(scan_data.items()):
        ax = axes[idx // cols][idx % cols]
        for col, color in colors.items():
            if df[col].notna().any():
                ax.plot(df["angle"], df[col], color=color, label=labels[col], lw=2, marker="o", ms=4)
        ax.set_title(mol_name, fontsize=11)
        ax.set_xlabel("Dihedral Angle (°)")
        ax.set_ylabel("ΔE (kcal/mol)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 360)

    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    plt.suptitle("SynergyFF — Torsion Profiles (Ref: B3LYP/6-31G*)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "torsion_profiles.png"), dpi=150, bbox_inches="tight")
    plt.close()

def plot_torsion_summary(df, output_path):
    if df.empty: return
    mae = {
        "MMFF":     df["err_mmff"].dropna().mean(),
        "UFF":      df["err_uff"].dropna().mean(),
        "ANI-2x":   df["err_ani"].dropna().mean(),
        "Ensemble": df["err_ens"].mean(),
    }
    mae = {k: v for k, v in mae.items() if pd.notna(v) and np.isfinite(v)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    colors_map = {"MMFF":"#4472C4","UFF":"#FF0000","ANI-2x":"#70AD47","Ensemble":"#FFC000"}
    lbls, vals = list(mae.keys()), list(mae.values())
    clrs = [colors_map.get(l, "#888") for l in lbls]
    bars = ax.bar(lbls, vals, color=clrs, alpha=0.85, edgecolor="black", width=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.03,
                f"{val:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"Torsion Barrier MAE vs QM (n={len(df)})")
    ax.set_ylabel("MAE (kcal/mol)")
    ax.set_ylim(0, max(vals) * 1.4 + 0.1)
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1]
    ref = df["ref"].values
    ax.scatter(ref, df["b_ani"].values, color="green", alpha=0.7, s=80, label=f"ANI-2x (MAE={mae.get('ANI-2x',0):.2f})")
    ax.scatter(ref, df["b_mmff"].values, color="#4472C4", alpha=0.7, s=80, marker="s", label=f"MMFF (MAE={mae.get('MMFF',0):.2f})")
    ax.scatter(ref, df["b_ens"].values, color="orange", alpha=0.9, s=100, marker="^", label=f"Ensemble (MAE={mae.get('Ensemble',0):.2f})")
    lim_max = max(ref.max(), df["b_ani"].max(), df["b_mmff"].max()) * 1.2 + 0.5
    ax.plot([0, lim_max], [0, lim_max], "k--", lw=1.5, label="Perfect")
    ax.set_xlabel("B3LYP/6-31G* Barrier (kcal/mol)")
    ax.set_ylabel("Predicted Barrier (kcal/mol)")
    ax.set_title("Torsion Barriers: Predicted vs Reference")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return mae

# ═══════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SynergyFF — Ensemble Force Field")
    parser.add_argument("--generate",  action="store_true", help="ORCA inputs + SPICE download")
    parser.add_argument("--torsion",   action="store_true", help="Test3: Torsion barriers")
    parser.add_argument("--spice",     action="store_true", help="SPICE benchmark")
    parser.add_argument("--benchmark", action="store_true", help="ORCA benchmark (Test1+2)")
    parser.add_argument("--summary",   action="store_true", help="Final combined report")
    parser.add_argument("--molecules", type=int, default=100, help="Max molecules for SPICE")
    args = parser.parse_args()

    if not any([args.generate, args.torsion, args.spice, args.benchmark, args.summary]):
        print("SynergyFF\n")
        print("  --generate   ORCA inputs + SPICE download")
        print("  --torsion    Test3: Torsion barriers (no ORCA needed)")
        print("  --spice      SPICE benchmark")
        print("  --benchmark  ORCA benchmark (Test1+2)")
        print("  --summary    Generate final report & graphs")
        sys.exit(0)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SynergyFF — Kalman-Weighted FF Ensemble                     ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print("║  Test1/2: B3LYP/6-31G* (ORCA)                                ║")
    print("║  Test3:   Torsion barriers (MMFF/UFF/ANI-2x)                 ║")
    print("║  SPICE:   ωB97M-D3BJ/def2-TZVPPD                             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    np.random.seed(42)
    torch.manual_seed(42)
    os.makedirs("./results", exist_ok=True)

    if args.generate:
        write_orca_inputs(ORCA_TEST1, "./orca_jobs/test1", n_confs=2, test_name="test1")
        write_orca_inputs(ORCA_TEST2, "./orca_jobs/test2", n_confs=2, test_name="test2")
        download_spice()

    if args.torsion:
        ani_model, converter = load_ani2x()
        if ani_model is not None:
            df_tor, k_tor, scan_data = run_torsion_benchmark(ani_model, converter)
            if not df_tor.empty:
                df_tor.to_csv("./results/torsion_results.csv", index=False)
                plot_torsion_profiles(scan_data, "./results")
                mae_tor = plot_torsion_summary(df_tor, "./results/torsion_summary.png")
                plot_kalman_evolution(k_tor, "./results/kalman_evolution_torsion.png")
                print(f"\n✅ Torsion Test Complete. MAE Ensemble: {mae_tor.get('Ensemble', 0):.3f} kcal/mol")

    if args.spice:
        ani_model, converter = load_ani2x()
        if ani_model is not None:
            df_spice, k_spice = run_spice_benchmark(ani_model, converter, args.molecules)
            if not df_spice.empty:
                df_spice.to_csv("./results/spice_results.csv", index=False)
                plot_benchmark(df_spice, "SPICE (ωB97M-D3BJ)", "./results/spice_benchmark.png")
                plot_qm_vs_synergy(df_spice, "./results/qm_vs_synergy_spice.png")
                plot_kalman_evolution(k_spice, "./results/kalman_evolution_spice.png")
                print("\n✅ SPICE Benchmark Complete.")

    if args.benchmark:
        ani_model, converter = load_ani2x()
        if ani_model is not None:
            for test_name, molecules, output_dir in [
                ("Test1_Small",    ORCA_TEST1, "./orca_jobs/test1"),
                ("Test2_DrugLike", ORCA_TEST2, "./orca_jobs/test2"),
            ]:
                if not os.path.exists(output_dir): continue
                df, kalman = run_orca_benchmark(molecules, output_dir, ani_model, converter, test_name)
                if not df.empty:
                    df.to_csv(f"./results/orca_{test_name}.csv", index=False)
                    plot_benchmark(df, f"ORCA {test_name}", f"./results/orca_{test_name}.png")
                    plot_qm_vs_synergy(df, f"./results/qm_vs_synergy_{test_name}.png")
                    plot_kalman_evolution(kalman, f"./results/kalman_evolution_{test_name}.png")

    if args.summary:
        print("Building Final Report...")
        all_results = {}
        for label, fname in [
            ("SPICE\n(ωB97M-D3BJ)",    "spice_results.csv"),
            ("ORCA\nTest1 (Small)",     "orca_Test1_Small.csv"),
            ("ORCA\nTest2 (Drug-like)", "orca_Test2_DrugLike.csv"),
        ]:
            path = os.path.join("./results", fname)
            if os.path.exists(path):
                df = pd.read_csv(path)
                mae = {
                    "MMFF":     df["err_mmff"].dropna().mean(),
                    "UFF":      df["err_uff"].dropna().mean(),
                    "ANI-2x":   df["err_ani"].mean(),
                    "Ensemble": df["err_ens"].mean(),
                }
                all_results[label] = (df, mae)

        tor_path = "./results/torsion_results.csv"
        if os.path.exists(tor_path):
            df = pd.read_csv(tor_path)
            mae = {
                "MMFF":     df["err_mmff"].dropna().mean(),
                "UFF":      df["err_uff"].dropna().mean(),
                "ANI-2x":   df["err_ani"].dropna().mean(),
                "Ensemble": df["err_ens"].mean(),
            }
            all_results["Torsion\nTest3"] = (df, mae)

        if all_results:
            plot_combined_summary(all_results, "./results/synergy_final_summary.png")
            plot_qm_vs_predicted(all_results, "./results/qm_vs_predicted.png")
            plot_kalman_trust_evolution(all_results, "./results/kalman_trust_evolution.png")
            compute_full_metrics(all_results)
            print("\n✅ All graphs saved in ./results folder.")
        else:
            print("  Run --torsion, --spice, and/or --benchmark first.")