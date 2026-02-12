# mcmc-hierarchical-logistic
MCMC for logistic regression with hierarchical priors.

This repository provides a reference implementation of an MCMC sampler for Bayesian logistic regression with hierarchical priors. It is intended to demonstrate Bayesian computation and posterior analysis in a compact, reproducible script.

## Repository contents
- `MCMC_logistic_regression_hierarchical.py` — main script (data simulation, sampling, diagnostics, and plots)

## How to run
```bash
python MCMC_logistic_regression_hierarchical.py
```

## Output
Generates standard MCMC diagnostics (trace plots, ACF), posterior summaries, and ESS.
