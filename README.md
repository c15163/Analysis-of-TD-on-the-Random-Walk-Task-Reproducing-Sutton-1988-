# Reproducing Sutton (1988): Analysis of TD(λ) on the Random Walk Task

This repository reproduces the classic temporal-difference learning experiments from Richard Sutton’s 1988 paper, **“Learning to Predict by the Methods of Temporal Differences.”**  
The goal of this project is to replicate Sutton’s figures for the 7-state random walk task, analyze the behavior of TD(λ) under different training schemes, and compare prediction accuracy across λ and α parameters.

All figures presented in the accompanying report are generated directly from the Python implementation included here.

---

## Project Structure

```
project_root/
│
├── TDLambda_random_walk.py      # Updated implementation with TDLambdaRandomWalk class
├── TD_Lambda-report.pdf         # Full analysis and reproduced figures
└── README.md                    # Documentation (this file)
```

---

## Background

TD(λ) is a family of temporal-difference learning algorithms that combines:

- Bootstrapping (as in TD learning)
- Multi-step returns (as in Monte-Carlo)
- Credit assignment controlled by the λ parameter  
  (λ = 0 behaves like TD(0), λ = 1 approximates Monte-Carlo)

Sutton’s 1988 paper introduced TD(λ) and demonstrated its properties using a simple 7-state random walk environment.  
This project faithfully reproduces those experiments.

---

## Environment: 7-State Random Walk

- States: A B C D E F G  
- Start at state **D**  
- Left/right transitions with equal probability  
- Terminal states:  
  - A → return = 0  
  - G → return = 1  

Each state is represented as a one-hot vector of length 7, following Sutton’s original setup.

---

## Experiments Reproduced

### Experiment 1 — Repeated Presentations until Convergence  
For a fixed learning rate (**α = 0.2**):

- Present 10 sequences repeatedly  
- Update weight vector after each batch  
- Stop when the weight vector stabilizes  
- Compute RMSE between learned weights and true values  
- Repeat 100 times for averaging  
- Produce RMSE vs λ curve (Figure 2 in report)

### Experiment 2 — Single Pass over Training Set  
For each combination of α and λ:

- Present 10 sequences only once (no convergence criterion)  
- Update weights after each episode  
- Compute RMSE after the single pass  
- Repeat 100 times  
- Produce α–λ performance curves (Figure 3)

### Best-α Analysis  
- Compute average RMSE across λ  
- Select best α  
- Plot full RMSE vs λ using this α (Figure 4)  
- Repeat excluding Widrow’s special cases λ = 0.8, 1.0 (Figure 5)

All figures exactly match the structure and intent of Sutton (1988).

---

## How to Run

Install dependencies:

```bash
pip install numpy matplotlib
```

Execute the experiment script:

```bash
python TDLambda_random_walk.py
```

Output figures:

```
figure2.png   # RMSE vs λ (convergence case)
figure3.png   # RMSE vs α for several λ
figure4.png   # Best α across all λ
figure5.png   # Best α excluding Widrow cases
```

---

## Key Results

- TD(λ) performance varies smoothly with λ under repeated presentations.  
- Under single-presentation learning, the interaction between learning rate (α) and λ is much more noticeable.  
- The optimal α differs depending on whether Widrow’s λ = 0.8 and λ = 1.0 cases are included.  
- The reproduced curves closely match those reported in Sutton (1988).

Full discussion and visualizations appear in **TD_Lambda-report.pdf**.

---

## Reference

Richard S. Sutton (1988). *Learning to Predict by the Methods of Temporal Differences.*  
Machine Learning 3, 9–44.

