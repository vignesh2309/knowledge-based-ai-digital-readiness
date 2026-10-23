# Knowledge-Based AI for Digital Transformation Readiness

**Research prototype:** *A Knowledge-Based AI Decision-Support Model for Predicting Digital Transformation Readiness from Enterprise Innovation Data.*

A compact, reproducible demonstration comparing (1) calibrated logistic regression, (2) interpretable knowledge-based rules, and (3) a fixed-weight hybrid. Six normalized indicators are governance, innovation investment, knowledge sharing, leadership, infrastructure, and workforce skills.

## Run

```bash
python -m pip install -r requirements.txt
python model.py --output results
```

Outputs: `results/metrics.csv`, `results/metrics.json`, and **two figures** (`figure1_framework.png`, `figure2_results.png`). No source dataset or credentials are required.

## Method

Synthetic enterprises are generated from correlated latent factors. A separate noisy outcome mechanism produces readiness labels. A stratified 75/25 train/test split is used; logistic regression is cross-validated for probability calibration on training data only. Knowledge rules use fixed expert-*inspired* weights and penalties, not empirically validated expert elicitation. The hybrid averages the two probabilities without fitting its weight on the test set. Compare held-out ROC AUC, accuracy, and Brier score.

## Scientific limitations

**Synthetic experiments do not establish external validity or generalization to real enterprises.** The outcome generator is a modeling assumption, not ground truth about organizations. Knowledge-rule weights, thresholds, and the 50/50 fusion are hypotheses. Do not claim hybrid superiority without supported results. A journal-ready study needs ethically collected enterprise data, an operational readiness definition, validation across organizations and time, uncertainty intervals, and sensitivity/fairness analyses.

## Minimal repository

- `model.py`: data generation, rules, training, evaluation, figures
- `requirements.txt`: dependencies
- `README.md`: reproduction and limitations

The accompanying short-communication PDF can be archived separately as a manuscript artifact. Results and figures are generated locally rather than committed as bulky binaries.
