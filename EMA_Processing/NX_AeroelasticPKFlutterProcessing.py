import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, simpledialog, messagebox
from dataclasses import dataclass
from typing import List, Dict, Optional


# -----------------------------
# ISA helper
# -----------------------------
@dataclass(frozen=True)
class ISAState:
    a: float  # m/s


def isa_speed_of_sound(alt_m: float) -> ISAState:
    gamma = 1.4
    R = 287.05287
    T0 = 288.15

    if alt_m < 0:
        alt_m = 0.0

    if alt_m <= 11000:
        T = T0 - 0.0065 * alt_m
    else:
        T = T0 - 0.0065 * 11000  # simplified isothermal above 11 km

    a = math.sqrt(gamma * R * T)
    return ISAState(a=a)


# -----------------------------
# Parsing SOL145 flutter tables
# -----------------------------
FLOAT = r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[Ee][+-]?\d+)?"

HEADER_RE = re.compile(
    r"FLUTTER\s+SUMMARY.*?MACH\s+NUMBER\s*=\s*(?P<mach>" + FLOAT + r")"
    r".*?DENSITY\s+RATIO\s*=\s*(?P<dr>" + FLOAT + r")",
    re.DOTALL | re.IGNORECASE
)

ROW_RE = re.compile(
    r"^\s*(?P<kfreq>" + FLOAT + r")\s+"
    r"(?P<invk>" + FLOAT + r")\s+"
    r"(?P<vel>" + FLOAT + r")\s+"
    r"(?P<damp>" + FLOAT + r")\s+"
    r"(?P<freq>" + FLOAT + r")\s+"
    r"(?P<eig_r>" + FLOAT + r")\s+"
    r"(?P<eig_i>" + FLOAT + r")\s*$"
)


def parse_flutter_summary(text: str) -> pd.DataFrame:
    """
    Returns tidy dataframe with columns:
      mach, density_ratio, block_id, velocity_mm_s, velocity_m_s, damping, frequency
    Each block_id corresponds to one FLUTTER SUMMARY block encountered.
    """
    rows: List[Dict] = []
    block_id = 0

    for m in HEADER_RE.finditer(text):
        block_id += 1
        mach = float(m.group("mach"))
        dr = float(m.group("dr"))

        chunk = text[m.end():m.end() + 25000]

        started = False
        for line in chunk.splitlines():
            r = ROW_RE.match(line.rstrip("\n"))
            if not r:
                if started:
                    break
                continue

            started = True
            vel_mm_s = float(r.group("vel"))
            rows.append({
                "mach": mach,
                "density_ratio": dr,
                "block_id": block_id,
                "velocity_mm_s": vel_mm_s,
                "velocity_m_s": vel_mm_s / 1000.0,
                "damping": float(r.group("damp")),
                "frequency": float(r.group("freq")),
            })

    if not rows:
        raise RuntimeError("No FLUTTER SUMMARY data found. Check the .f06 content.")

    df = pd.DataFrame(rows)
    df = df.sort_values(["density_ratio", "mach", "block_id", "velocity_m_s"]).reset_index(drop=True)
    return df


# -----------------------------
# Mode identification (based on first entry)
# -----------------------------
def mode_reference_frequencies(df: pd.DataFrame, mach: float, dr: float) -> pd.DataFrame:
    """
    For the chosen (mach, dr), compute each block's first-entry frequency f0.
    (First row of that block after sorting by velocity)
    """
    sel = df[(df["mach"] == mach) & (df["density_ratio"] == dr)].copy()
    if sel.empty:
        return pd.DataFrame(columns=["block_id", "f0", "v0"])

    sel = sel.sort_values(["block_id", "velocity_m_s"])
    first = sel.groupby("block_id", as_index=False).first()
    out = first[["block_id", "frequency", "velocity_m_s"]].rename(
        columns={"frequency": "f0", "velocity_m_s": "v0"}
    )
    return out.sort_values("f0").reset_index(drop=True)


def block_f0_lookup(df: pd.DataFrame, mach: float, dr: float) -> Dict[int, float]:
    mode_table = mode_reference_frequencies(df, mach, dr)
    return {int(r["block_id"]): float(r["f0"]) for _, r in mode_table.iterrows()}


def ask_mode_selection(root: tk.Tk, mode_table: pd.DataFrame) -> Optional[List[int]]:
    """
    Returns:
      - None  => plot all modes
      - list of block_ids => plot only selected modes
    """
    if mode_table.empty:
        messagebox.showwarning(
            "No modes found",
            "No mode tracks found for the selected Mach and density ratio."
        )
        return None

    lines = []
    for _, row in mode_table.iterrows():
        lines.append(f"  {int(row['block_id'])}: f0={row['f0']:.6g} Hz (v0={row['v0']:.6g} m/s)")

    msg = (
        "Mode tracks detected for selected Mach & density ratio.\n"
        "Each entry uses the FIRST table row in that mode track as reference (f0).\n\n"
        "Available modes (block_id: f0):\n"
        + "\n".join(lines)
        + "\n\n"
        "Type block_ids to plot, comma-separated (e.g. 1,3,7)\n"
        "Or type ALL to plot everything."
    )

    s = simpledialog.askstring("Select Modes", msg, parent=root)
    if s is None:
        return None

    s = s.strip().upper()
    if s == "" or s == "ALL":
        return None

    try:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        ids = [int(p) for p in parts]
    except ValueError:
        messagebox.showerror("Invalid input", "Please enter comma-separated integers or ALL.")
        return ask_mode_selection(root, mode_table)

    available = set(mode_table["block_id"].astype(int).tolist())
    chosen = [i for i in ids if i in available]
    missing = [i for i in ids if i not in available]

    if not chosen:
        messagebox.showerror("No valid modes selected", "None of the entered block_ids exist in the list.")
        return ask_mode_selection(root, mode_table)

    if missing:
        messagebox.showwarning(
            "Some modes not found",
            f"These block_ids were not found and will be ignored:\n{missing}\n\nProceeding with: {chosen}"
        )

    return chosen


# -----------------------------
# Plotting
# -----------------------------
def plot_velocity(
    df: pd.DataFrame,
    mach: float,
    dr: float,
    f0_map: Optional[Dict[int, float]] = None,
    mode_ids: Optional[List[int]] = None
):
    """
    Velocity–Damping and Velocity–Frequency for selected Mach & density ratio.
    Optionally filter by mode_ids. Legend shows f0 when available.
    Velocity is plotted in m/s. Frequency axis labeled in Hz.
    """
    sel = df[(df.mach == mach) & (df.density_ratio == dr)].copy()
    if mode_ids is not None:
        sel = sel[sel["block_id"].isin(mode_ids)]

    if sel.empty:
        raise RuntimeError("No data after applying Mach/DR/mode filters.")

    def label_for_block(bid: int) -> str:
        f0 = f0_map.get(bid) if f0_map else None
        return f"f₀ = {f0:.3g} Hz" if f0 is not None else f"block {bid}"

    # Velocity–Damping
    plt.figure()
    for bid, g in sel.groupby("block_id"):
        g = g.sort_values("velocity_m_s")
        plt.plot(g.velocity_m_s, g.damping, label=label_for_block(int(bid)))
    plt.axhline(0)
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Damping")
    plt.title(f"Velocity–Damping @ Mach {mach:.6g}, DR {dr:.6g}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)

    # Velocity–Frequency
    plt.figure()
    for bid, g in sel.groupby("block_id"):
        g = g.sort_values("velocity_m_s")
        plt.plot(g.velocity_m_s, g.frequency, label=label_for_block(int(bid)))
    plt.xlabel("Velocity (m/s)")
    plt.ylabel("Frequency (Hz)")
    plt.title(f"Velocity–Frequency @ Mach {mach:.6g}, DR {dr:.6g}")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)

    plt.show()


def nearest_block_by_f0_in_table(mode_table: pd.DataFrame, target_f0: float) -> Optional[int]:
    """
    Given a mode_table with columns [block_id, f0, v0] for ONE (mach, dr),
    return the block_id whose f0 is closest to target_f0.
    """
    if mode_table.empty:
        return None
    idx = int(np.argmin(np.abs(mode_table["f0"].to_numpy() - target_f0)))
    return int(mode_table.iloc[idx]["block_id"])


def plot_mach_damping_selected_modes(
    df: pd.DataFrame,
    dr: float,
    alt_m: float,
    selected_f0s: List[float],
):
    """
    Mach–Damping at altitude alt_m using ISA only, for ALL selected modes together.

    IMPORTANT: mode tracking across Mach is done by f0 matching per Mach slice
    (because block_id is not stable across Mach in the F06 parsing).
    """
    isa = isa_speed_of_sound(alt_m)
    a_m_s = isa.a

    slice_dr = df[df.density_ratio == dr].copy()
    if slice_dr.empty:
        raise RuntimeError("No data for selected density ratio.")

    machs = sorted(slice_dr.mach.unique())

    # Debug ranges (global)
    vmin_data = float(slice_dr.velocity_m_s.min())
    vmax_data = float(slice_dr.velocity_m_s.max())
    vmin_target = float(min(machs) * a_m_s)
    vmax_target = float(max(machs) * a_m_s)

    plt.figure()
    plotted_any = False

    for f0_target in selected_f0s:
        out_m, out_d = [], []

        for M in machs:
            v_target = M * a_m_s  # m/s

            # For this Mach slice, build a mode table and pick the closest f0
            mode_table_M = mode_reference_frequencies(df, M, dr)
            bid_M = nearest_block_by_f0_in_table(mode_table_M, f0_target)
            if bid_M is None:
                continue

            gM = slice_dr[(slice_dr.mach == M) & (slice_dr.block_id == bid_M)].copy()
            if gM.empty:
                continue

            gM = gM.sort_values("velocity_m_s")
            vmin, vmax = float(gM.velocity_m_s.min()), float(gM.velocity_m_s.max())

            if vmin <= v_target <= vmax:
                d = float(np.interp(v_target, gM.velocity_m_s.to_numpy(), gM.damping.to_numpy()))
                out_m.append(M)
                out_d.append(d)

        if len(out_m) >= 2:
            plt.plot(out_m, out_d, label=f"f₀≈{f0_target:.3g} Hz")
            plotted_any = True

    if not plotted_any:
        plt.close()
        messagebox.showwarning(
            "No curves plotted",
            "Even though global velocity ranges overlap, none of the selected f₀ tracks\n"
            "had >=2 Mach points where V(M)=M*a(alt) fell inside their per-Mach sweep.\n\n"
            f"Altitude: {alt_m:.0f} m  ->  a = {a_m_s:.3g} m/s\n"
            f"Target V range over Machs in file: [{vmin_target:.3g}, {vmax_target:.3g}] m/s\n"
            f"Data velocity range in file (this DR): [{vmin_data:.3g}, {vmax_data:.3g}] m/s\n\n"
            "Try a different altitude or select different modes."
        )
        return

    plt.axhline(0)
    plt.xlabel("Mach")
    plt.ylabel("Damping")
    plt.title(f"Mach–Damping @ altitude {alt_m:.0f} m, DR {dr:.6g} (ISA)")
    plt.grid(True)
    plt.legend(fontsize=8, ncol=2)
    plt.show()




# -----------------------------
# GUI main
# -----------------------------
def main():
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Nastran .f06 text file",
        filetypes=[("Nastran output", "*.f06 *.txt"), ("All files", "*.*")]
    )
    if not file_path:
        return

    with open(file_path, "r", errors="ignore") as f:
        text = f.read()

    df = parse_flutter_summary(text)

    machs = sorted(df.mach.unique())
    drs = sorted(df.density_ratio.unique())

    mach_in = simpledialog.askfloat(
        "Select Mach",
        f"Available Mach values:\n{machs}\n\nEnter desired Mach (nearest will be used):"
    )
    if mach_in is None:
        return
    mach = min(machs, key=lambda x: abs(x - mach_in))

    dr_in = simpledialog.askfloat(
        "Select Density Ratio",
        f"Available density ratios:\n{drs}\n\nEnter desired density ratio (nearest will be used):"
    )
    if dr_in is None:
        return
    dr = min(drs, key=lambda x: abs(x - dr_in))

    # Mode info for this Mach+DR
    mode_table = mode_reference_frequencies(df, mach, dr)
    f0_map = block_f0_lookup(df, mach, dr)

    if mode_table.empty or not f0_map:
        messagebox.showerror("No modes", "No f₀/mode data available for this Mach & density ratio.")
        return

    # Choose modes to include
    if messagebox.askyesno("Mode selection", "Do you want to select specific modes to plot?"):
        mode_ids = ask_mode_selection(root, mode_table)  # None => ALL
    else:
        mode_ids = None

    if mode_ids is None:
        mode_ids = sorted(mode_table["block_id"].astype(int).tolist())

    # Velocity plots (all selected modes)
    plot_velocity(df, mach, dr, f0_map=f0_map, mode_ids=mode_ids)

    # Mach–Damping (all selected modes)
    alt = simpledialog.askfloat("Altitude", "Enter altitude [m]:", minvalue=0)
    if alt is None:
        return
    selected_f0s = [f0_map[i] for i in mode_ids if i in f0_map]
    plot_mach_damping_selected_modes(df, dr, alt, selected_f0s=selected_f0s)


if __name__ == "__main__":
    main()





