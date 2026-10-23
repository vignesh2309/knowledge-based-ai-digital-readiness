"""Reproducible knowledge-guided digital transformation readiness experiment.

Synthetic data are illustrative, NOT evidence of real-world predictive validity.
Run: python model.py --output results
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["governance", "innovation", "knowledge_sharing",
            "leadership", "infrastructure", "skills"]
WEIGHTS = np.array([.20, .18, .15, .19, .15, .13])
SEED = 42


def generate_data(n=1500, seed=SEED):
    """Simulate correlated organizational indicators and a noisy readiness label."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(n, 2))
    loadings = np.array([[.9, .1], [.2, .9], [.6, .5],
                         [.8, .3], [.7, .3], [.4, .8]])
    z = latent @ loadings.T + rng.normal(scale=.75, size=(n, 6))
    x = 1 / (1 + np.exp(-z))
    # Outcome-generating mechanism is separate from the knowledge rules.
    logit = -1.1 + 2.1 * x[:, 0] + 1.3 * x[:, 1] + 1.1 * x[:, 3] \
            + .9 * x[:, 5] - .7 * (x[:, 4] < .3) + rng.normal(scale=.45, size=n)
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame(x, columns=FEATURES), y


def knowledge_score(x):
    """Expert-inspired transparent rules; weights/thresholds are hypotheses."""
    a = np.asarray(x)
    score = a @ WEIGHTS
    penalty = .08 * (a[:, 0] < .35) + .06 * (a[:, 5] < .35)
    return np.clip(score - penalty, 0, 1)


def fit_models(x_train, y_train, seed=SEED):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    ml = CalibratedClassifierCV(
        make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        method="sigmoid", cv=cv
    )
    ml.fit(x_train, y_train)
    return ml


def evaluate(y, probability):
    return {
        "roc_auc": round(float(roc_auc_score(y, probability)), 4),
        "accuracy": round(float(accuracy_score(y, probability >= .5)), 4),
        "brier": round(float(brier_score_loss(y, probability)), 4),
    }


def experiment(output="results", n=1500, seed=SEED):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    x, y = generate_data(n, seed)
    xt, xv, yt, yv = train_test_split(
        x, y, test_size=.25, random_state=seed, stratify=y
    )
    ml = fit_models(xt, yt, seed)
    p_ml = ml.predict_proba(xv)[:, 1]
    p_kb = knowledge_score(xv)
    # Predeclared 50/50 fusion, not tuned on held-out test labels.
    p_hybrid = .5 * p_ml + .5 * p_kb
    scores = {
        "ML": evaluate(yv, p_ml),
        "Knowledge rules": evaluate(yv, p_kb),
        "Hybrid": evaluate(yv, p_hybrid),
    }
    pd.DataFrame.from_dict(scores, orient="index").rename_axis("model").to_csv(
        out / "metrics.csv"
    )
    (out / "metrics.json").write_text(json.dumps({
        "seed": seed, "n": n, "test_n": len(yv),
        "synthetic": True, "scores": scores
    }, indent=2))
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.axis("off")
    nodes = [("Innovation\nindicators", .10), ("Knowledge\nrules", .36),
             ("Calibrated\nclassifier", .62), ("Decision\nsupport", .88)]
    for label, xpos in nodes:
        ax.text(xpos, .5, label, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=.65", facecolor="white",
                          edgecolor="black"))
    for a, b in zip(nodes[:-1], nodes[1:]):
        ax.annotate("", xy=(b[1]-.09, .5), xytext=(a[1]+.09, .5),
                    arrowprops=dict(arrowstyle="->"))
    fig.savefig(out / "figure1_framework.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(scores)
    vals = [scores[name]["roc_auc"] for name in names]
    ax.bar(names, vals)
    ax.set_ylabel("Held-out ROC AUC")
    ax.set_ylim(0, 1)
    ax.set_title("Synthetic demonstration (not external validation)")
    for i, v in enumerate(vals):
        ax.text(i, v+.02, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(out / "figure2_results.png", dpi=180)
    plt.close(fig)
    print(json.dumps(scores, indent=2))
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results")
    parser.add_argument("--n", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    experiment(args.output, args.n, args.seed)
