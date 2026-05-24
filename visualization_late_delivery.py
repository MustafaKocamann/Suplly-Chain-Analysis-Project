# Late Delivery — Model Visualization Pipeline
# Supply Chain Project

from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns
warnings.filterwarnings("ignore")

PRIMARY   = '#1B3A6B'
ACCENT    = '#2E86AB'
HIGHLIGHT = '#E63946'
SUCCESS   = '#2DC653'
WARN      = '#F4A261'
LIGHT     = '#48CAE4'
BG        = '#F8F9FA'

plt.rcParams.update({
    'font.family'    : 'DejaVu Sans',
    'font.size'      : 11,
    'axes.titlesize' : 13,
    'axes.labelsize' : 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
})


# ══════════════════════════════════════════════════════════════════
#  YARDIMCI — En iyi modeli bul
# ══════════════════════════════════════════════════════════════════
def _get_best_model(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda r: r["late_delivery"]["roc_auc"]
                      if r["late_delivery"]["roc_auc"] is not None else 0
    )


# ══════════════════════════════════════════════════════════════════
#  1. CONFUSION MATRIX — En iyi model
# ══════════════════════════════════════════════════════════════════
def plot_confusion_matrix(results: list[dict],
                          model_name: str = None,
                          save: bool = True) -> None:
    """
    # Confusion Matrix — Raw count + Normalized (%) yan yana.
    model_name=None #ise en iyi model otomatik seçilir.
    """
    if model_name is None:
        result = _get_best_model(results)
        model_name = result["model_name"]
    else:
        result = next(
            (r for r in results if r["model_name"] == model_name), None)
        if result is None:
            print(f"Model '{model_name}' bulunamadı.")
            return

    cm      = result["late_delivery"]["confusion_matrix"]
    cm_pct  = cm.astype(float) / cm.sum() * 100
    total   = cm.sum()
    accuracy= np.trace(cm) / total
    labels  = ["On Time / Advance", "Late Delivery"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    fig.suptitle(
        f'Confusion Matrix — {model_name}  (Best Model)\n'
        f'Late Delivery Prediction  |  Accuracy: {accuracy:.4f}',
        fontsize=16, fontweight='bold', color=PRIMARY, y=1.02)

    cell_colors = np.array([
        ["#D4EFDF", "#FADBD8"],
        ["#FADBD8", "#D4EFDF"]
    ])

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ['d', '.1f'],
        ['Raw Count', 'Normalized (%)']
    ):
        ax.set_facecolor(BG)
        for i in range(2):
            for j in range(2):
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=cell_colors[i][j],
                    edgecolor='white', linewidth=2, zorder=1))

        for i in range(2):
            for j in range(2):
                val   = data[i, j]
                color = PRIMARY if cell_colors[i][j] == "#D4EFDF" \
                        else HIGHLIGHT
                suffix = '%' if fmt == '.1f' else ''
                ax.text(j, i, f'{val:{fmt}}{suffix}',
                        ha='center', va='center',
                        fontsize=16, fontweight='bold',
                        color=color, zorder=3)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10,
                            rotation=90, va='center')
        ax.set_xlabel('Predicted Label', fontsize=11, color='#444')
        ax.set_ylabel('True Label',      fontsize=11, color='#444')
        ax.set_title(title, fontsize=12, fontweight='bold',
                     color=PRIMARY, pad=10)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)
        ax.spines[['top','right','bottom','left']].set_visible(False)

        for (i, j), lbl in [((0,0),'TN'), ((0,1),'FP'),
                              ((1,0),'FN'), ((1,1),'TP')]:
            ax.text(j - 0.45, i - 0.45, lbl,
                    fontsize=8, color='#888', va='top', ha='left')

    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) \
                if (precision + recall) > 0 else 0

    kpi = (f"TP:{tp:,}  |  TN:{tn:,}  |  FP:{fp:,}  |  FN:{fn:,}"
           f"  |  Precision:{precision:.4f}"
           f"  |  Recall:{recall:.4f}  |  F1:{f1:.4f}")
    fig.text(0.5, -0.02, kpi, ha='center', fontsize=10,
             color=PRIMARY, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                       edgecolor=PRIMARY, linewidth=1))
    fig.text(0.99, -0.07,
             'Source: Supply Chain Dataset  |  Senior Data Analytics Team',
             ha='right', fontsize=8, color='#888', style='italic')

    plt.tight_layout(pad=2.5)
    plt.show()




# ══════════════════════════════════════════════════════════════════
#  NOTEBOOK USAGE
# ══════════════════════════════════════════════════════════════════
# from visualization_late_delivery import (
#     plot_confusion_matrix,
#     plot_roc_curve_best,
#     plot_animated_bar_race,
#     plot_bubble_scatter,
# )
#
# # 1. Confusion Matrix — en iyi model otomatik seçilir
# plot_confusion_matrix(results)
#
# # 2. ROC-AUC — sadece en iyi model
# plot_roc_curve_best(results, y_true=output.yl_test)
#
# # 3. Animated Bar Race — ROC-AUC metriği
# plot_animated_bar_race(late_cv_df, metric='ROC-AUC')
#
# # 4. Bubble Scatter
# plot_bubble_scatter(late_cv_df)