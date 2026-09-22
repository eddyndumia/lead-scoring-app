"""Generates the capacity/precision-recall figure referenced in the README."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"
FIG_DIR = REPORTS_DIR / "figures"


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS_DIR / "capacity_thresholds.csv")

    fig, ax1 = plt.subplots(figsize=(7.5, 4.5))
    ax1.plot(df["capacity_pct"], df["precision"], marker="o", color="#2980b9", label="Precision")
    ax1.set_xlabel("Sales team capacity (% of leads contacted)")
    ax1.set_ylabel("Precision", color="#2980b9")
    ax1.tick_params(axis="y", labelcolor="#2980b9")

    ax2 = ax1.twinx()
    ax2.plot(df["capacity_pct"], df["recall"], marker="s", color="#c0392b", label="Recall")
    ax2.set_ylabel("Recall", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")

    fig.suptitle("Precision / recall tradeoff by sales team capacity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "capacity_tradeoff.png", dpi=120)
    plt.close(fig)
    print(f"Saved figure -> {FIG_DIR / 'capacity_tradeoff.png'}")


if __name__ == "__main__":
    main()
