import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional
from loguru import logger


class GBVisualizer:
    @staticmethod
    def plot_feature_importance(importance, save_path=None):
        plt.figure(figsize=(8, 5))
        sns.barplot(x=list(importance.values()), y=list(importance.keys()), palette="Oranges_r")
        plt.xlabel("Importance")
        plt.title("Feature Importance — Gradient Boosting")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    @staticmethod
    def plot_confusion_matrix(cm, class_names, save_path=None):
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted"); plt.ylabel("Actual")
        plt.title("Confusion Matrix — Gradient Boosting")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
