#%% Bayesian Logistic Regression with Metropolis-Hastings
# 
# This file contains the scaffolding for implementing Bayesian logistic regression
# using the Metropolis-Hastings algorithm. Your task is to fill in the missing
# parts marked with TODO comments and complete the exercises.
#
# Learning Objectives:
# 1. Understand the Metropolis-Hastings algorithm
# 2. Implement log-likelihood and log-prior functions
# 3. Create MCMC diagnostics and visualizations
# 4. Interpret Bayesian inference results

#%% Import libraries and setup
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit  # sigmoid function
from scipy.stats import norm, multivariate_normal, pearsonr, gaussian_kde
import pandas as pd

sns.set(style="whitegrid")

#%% Generate simulated data
# This creates a synthetic dataset for logistic regression
np.random.seed(42)
n = 100  # number of observations
p = 3    # number of parameters (intercept + 2 predictors)
X = np.random.randn(n, 2)  # predictor variables
X = np.hstack([np.ones((n, 1)), X])  # Add intercept term
true_beta = np.array([-0.5, 1.0, -1.0])  # true parameter values
logits = X @ true_beta  # linear combination
probs = expit(logits)   # apply sigmoid to get probabilities
y = np.random.binomial(1, probs)  # generate binary outcomes

print(f"Data shape: X = {X.shape}, y = {y.shape}")
print(f"True parameters: {true_beta}")
print(f"Proportion of 1s: {np.mean(y):.3f}")

#%% Visualize the data
df = pd.DataFrame(X[:, 1:], columns=["x1", "x2"])
df["y"] = y
sns.scatterplot(data=df, x="x1", y="x2", hue="y", palette="Set1")
plt.title("Simulated logistic regression data")
plt.show()

#%% Define likelihood and prior functions
# ---------- log-likelihood ----------
def log_likelihood(theta, X, y):
    eta = X @ theta
    return np.sum(y * eta - np.logaddexp(0.0, eta))                                               # stable logistic loglik

# ---------- log-prior ----------
def log_prior_theta(theta, mu, sigma2):
    return -0.5 / sigma2 * np.sum((theta - mu) ** 2)                                              # const dropped

def log_posterior_theta(theta, X, y, mu, sigma2):
    return log_likelihood(theta, X, y) + log_prior_theta(theta, mu, sigma2)

# ---------- Metropolis-within-Gibbs ----------
def mwg_sampler_A(
    X, y, mu0, k, a, b,
    n_samples=10000, burn=2000,
    proposal_sd=0.15, seed=None,
    init_theta=None, init_mu=None, init_sigma2=None
):
    """
      θ | μ,σ² ~ N(μ, σ² I)
      μ | σ²   ~ N(μ0, σ²/k I)
      σ²       ~ Inv-Gamma(a, b)   [density ∝ (σ²)^-(a+1) exp(-b/σ²)]
    """

    np.random.seed(seed)  

    n, p = X.shape

    # init
    theta  = np.zeros(p) 
    mu     = np.zeros(p) 
    sigma2 = 1.0       

    thetas = np.empty((n_samples, p))
    mus    = np.empty((n_samples, p))
    sig2s  = np.empty(n_samples)
    accepts = 0

    logp_curr = log_posterior_theta(theta, X, y, mu, sigma2)

    for t in range(n_samples):
        # 1) RW-MH: θ | μ, σ², data
        prop = theta + np.random.normal(scale=proposal_sd, size=p)
        logp_prop = log_posterior_theta(prop, X, y, mu, sigma2)
        if np.log(np.random.uniform()) < (logp_prop - logp_curr):
            theta, logp_curr = prop, logp_prop
            accepts += 1

        # 2) Gibbs: μ | θ, σ²
        post_var  = sigma2 / (1.0 + k)                 
        post_mean = (theta + k * mu0) / (1.0 + k)      
        mu = post_mean + np.sqrt(post_var) * np.random.normal(size=p)

        # 3) Gibbs: σ² | θ, μ
        a_post = a + p
        b_post = b + 0.5 * (np.sum((theta - mu) ** 2) + k * np.sum((mu - mu0) ** 2))
        sigma2 = 1.0 / np.random.gamma(shape=a_post, scale=1.0 / b_post)  

        thetas[t] = theta
        mus[t]    = mu
        sig2s[t]  = sigma2

    acc_rate = accepts / n_samples
    sl = slice(burn, None)

    return {
        # post-burn
        "theta":  thetas[sl],
        "mu":     mus[sl],
        "sigma2": sig2s[sl],
        "acc_rate": acc_rate,
        # full chain（including burn-in）
        "theta_all":  thetas,
        "mu_all":     mus,
        "sigma2_all": sig2s,
    }

#%% Run the sampler
print("Running sampler...")

p   = X.shape[1]
mu0 = np.zeros(p)
k, a, b = 0.1, 2.0, 2.0

out = mwg_sampler_A(
    X, y, mu0, k, a, b,
    n_samples=15000, burn=5000,
    proposal_sd=0.15,
    seed=100
)

theta_samples = out["theta"]
mu_samples    = out["mu"]
sig2_samples  = out["sigma2"]
acc_rate      = out["acc_rate"]

print(f"theta samples: {theta_samples.shape}, mu samples: {mu_samples.shape}, "
      f"sigma2 samples: {sig2_samples.shape}")
print(f"Acceptance rate (theta MH): {acc_rate:.3f}")


#%% Exercise: Create trace plots
# TODO: Create trace plots for each parameter
# Hint: Use subplots to create 3 plots (one for each parameter)
param_names = ['Intercept', 'θ1', 'θ2']

# YOUR CODE HERE:
# Create a figure with 3 subplots (3 rows, 1 column)
# For each parameter:
#   - Plot the trace (all_draws[:, i])
#   - Add a vertical line at the burn-in point
#   - Add title, labels, and legend


chain = out["theta_all"]                                   
burn = 5000           
param_names = ['Intercept', 'θ1', 'θ2']

iters = np.arange(15000)
fig, axes = plt.subplots(chain.shape[1], 1, figsize=(10, 6), sharex=True)
axes = np.atleast_1d(axes)

for j, ax in enumerate(axes):
    ax.plot(iters, chain[:, j], lw=0.8)
    ax.axvline(x=burn, ls='--', lw=1.0, color='k')         
    ax.set_ylabel(param_names[j])

axes[-1].set_xlabel('Iteration')
plt.tight_layout()
plt.show()

#%% Exercise: Create autocorrelation plots
# TODO: Create autocorrelation plots for each parameter
# you can either use pearsonr and the formula from slide 10 section 7(wk5) of the notes or the  
# autocorr plot function from matplotlib.pyplot or alternative python libraries


from statsmodels.graphics.tsaplots import plot_acf

# post-burn 
chain = out["theta"]                                                                     # shape: (n_keep, p)
names = ['Intercept','θ1', 'θ2']
max_lag = 100

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
axes = np.atleast_1d(axes)

for j, ax in enumerate(axes):
    plot_acf(chain[:, j], lags=max_lag, ax=ax, zero=False)
    ax.set_title(names[j])

axes[-1].set_xlabel("Lag")
plt.tight_layout()
plt.show()


#%% Exercise: Create posterior distribution plots
# TODO: Create histograms and kernel density plots for each parameter
# 
# This exercise will help you visualize the posterior distributions of your parameters.
# You'll create comprehensive plots that show:
# 1. Histogram of the posterior samples
# 2. Kernel density estimate 
# 3. True parameter values
# 4. Posterior mean and 95% credible intervals
#
# Instructions:
# - Create a figure with 1 row and 3 columns (one subplot per parameter)
# - For each parameter, create a histogram with density=True
# - Add a kernel density estimate using scipy.stats.gaussian_kde
# - Add vertical lines for true values, posterior mean, and credible intervals
# - Use different colors and styles to distinguish the elements
#


    # TODO: Create a comprehensive posterior plot
    
    # YOUR CODE HERE:
    # Step 1: Create histogram
    # ax.hist(post_samples[:, i],...)
    
    # Step 2: Create kernel density estimate
    # kde = gaussian_kde(?)
    # x_range = np.linspace(?, ?, 100)  # Create x values for smooth curve
    # ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
    
    # Step 3: Add true parameter value
    # ax.axvline(x=?, color='green', label=f'True {name}')
    
    # Step 4: Add posterior mean and credible interval
    # mean_val = np.mean(?)
    # ci_95 = np.percentile(?, [2.5, 97.5])  # 2.5th and 97.5th percentiles
    # ax.axvline(x=?, color='orange', linewidth=2, label=f'Mean: {mean_val:.3f}')
    # ax.axvspan(?, ?, alpha=0.2, color='orange', label=f'95% CI: [{ci_95[0]:.3f}, {ci_95[1]:.3f}]')
    
    # Step 5: Add labels and formatting


# post burn
post = theta_samples                           # shape: (n_keep, p)
names = ['Intercept', 'θ1', 'θ2']

true_vals = np.array([-0.5, 1.0, -1.0])

fig, axes = plt.subplots(1, post.shape[1], figsize=(15, 5), sharex=False)


for j, ax in enumerate(axes):
    xj = post[:, j]
    mean_j = xj.mean()
    ci_lo, ci_hi = np.percentile(xj, [2.5, 97.5])

    # 1) histgram
    ax.hist(xj, bins=40, density=True, alpha=0.35, color='tab:blue', label='Histogram')

    # 2) KDE
    kde = gaussian_kde(xj)
    x_min, x_max = mean_j - 4*np.std(xj), mean_j + 4*np.std(xj)
    xs = np.linspace(x_min, x_max, 300)
    ax.plot(xs, kde(xs), lw=2, label='KDE')

    # 3) true value
    ax.axvline(true_vals[j], color='green', lw=2, ls='--', label='True')

    # 4) posterior mean & 95% CI
    ax.axvline(mean_j, color='tab:red', lw=2, label=f'Mean: {mean_j:.3f}')
    ax.axvspan(ci_lo, ci_hi, color='orange', alpha=0.18, label=f'95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]')
    ax.axvline(ci_lo, color='orange', lw=1)
    ax.axvline(ci_hi, color='orange', lw=1)

    # 5) lables
    ax.set_title(f'Posterior: {names[j]}')
    ax.set_xlabel('Parameter value')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', frameon=False)

plt.tight_layout()
plt.show()


#%% Exercise: Compute summary statistics
# TODO: Compute and display summary statistics for each parameter
print("\n" + "="*60)
print("POSTERIOR SUMMARIES")
print("="*60)

print(f"{'Parameter':<12} {'Mean':<10} {'Std':<10} {'95% CI':<20} {'True':<10}")
print("-" * 60)

#for i, name in enumerate(param_names):
    # TODO: Compute statistics for this parameter
    # mean_val = ?
    # std_val = ?
    # ci_95 = ?
    
    # YOUR CODE HERE:
#    mean_val = None  # Replace with your implementation
#    std_val = None   # Replace with your implementation
#    ci_95 = None     # Replace with your implementation
    
#    print(f"{name:<12} {mean_val:<10.3f} {std_val:<10.3f} "
#          f"[{ci_95[0]:<8.3f}, {ci_95[1]:<8.3f}] {true_beta[i]:<10.3f}")

for i, name in enumerate(param_names):
    x = post[:, i]
    mean_val = x.mean()
    std_val  = x.std(ddof=1)                        
    ci_95    = np.percentile(x, [2.5, 97.5])        

   
    true_val = true_beta[i] 

    print(f"{name:<12} {mean_val:<10.3f} {std_val:<10.3f} "
          f"[{ci_95[0]:<8.3f}, {ci_95[1]:<8.3f}] {true_val:<10.3f}")
    
    
    
    # ========== 3 chains ==========
seeds = [100, 200, 300]  
inits = [
    dict(init_theta=np.zeros(p),     init_mu=mu0,         init_sigma2=1),
    dict(init_theta=mu0 + 2*np.ones(p), init_mu=mu0+1.0,  init_sigma2=2.0),
    dict(init_theta=mu0 - 2*np.ones(p), init_mu=mu0-1.0,  init_sigma2=1.5),
]

outs = [
    mwg_sampler_A(
        X, y, mu0, k, a, b,
        n_samples=15000, burn=5000, proposal_sd=0.15,
        seed=s, **ini
    )
    for s, ini in zip(seeds, inits)
]
chains = [o["theta_all"] for o in outs]   
burn   = 5000
param_names = ['Intercept', 'θ1', 'θ2']

iters = np.arange(chains[0].shape[0])
colors = ['tab:blue', 'tab:orange', 'tab:green']

fig, axes = plt.subplots(len(param_names), 1, figsize=(10, 6), sharex=True)
axes = np.atleast_1d(axes)

for j, ax in enumerate(axes):
    for c, ch in zip(colors, chains):
        ax.plot(iters, ch[:, j], lw=0.8, color=c)
    ax.axvline(burn, ls='--', lw=1.0, color='k')
    ax.set_ylabel(param_names[j])

axes[-1].set_xlabel('Iteration')

axes[-1].legend([f'Chain {i+1}' for i in range(len(chains))],
                loc='upper right', frameon=False)
plt.tight_layout()
plt.show()


# ============================================================
# Compute Effective Sample Size (ESS) for each parameter
# ============================================================

def autocorrelation_1d(x, max_lag=None):
    """
    Autocorrelation rho(k) for k = 0,1,...,max_lag for a 1D chain x.
    Returns an array acf where acf[0] = 1, acf[1] = rho(1), etc.
    """
    x = np.asarray(x)
    x_centered = x - np.mean(x)
    n = len(x_centered)

    # choose max_lag
    if (max_lag is None) or (max_lag > n - 1):
        max_lag = n - 1

    # autocovariance via correlate
    # full length is 2n-1; we keep only nonnegative lags
    acov_full = np.correlate(x_centered, x_centered, mode='full')
    acov = acov_full[n-1:]          # length n, lag 0..n-1
    acov = acov / n                 # scale ~variance

    var0 = acov[0]
    acf = acov[:max_lag+1] / var0   # normalize so acf[0] = 1
    return acf


def effective_sample_size(x, max_lag=1000):
    """
    ESS = N / (1 + 2 * sum_{k>=1} rho(k))
    """
    x = np.asarray(x)
    n = len(x)

    if (max_lag is None) or (max_lag > n - 1):
        max_lag = n - 1

    acf = autocorrelation_1d(x, max_lag=max_lag)

    tail_sum = 0.0
    # skip k=0 (that's 1 by definition), start from lag=1
    for k in range(1, len(acf)):
        if acf[k] <= 0:
            break
        tail_sum += acf[k]

    ess = n / (1.0 + 2.0 * tail_sum)
    return ess


post_chains = [o["theta"] for o in outs]   # list of arrays, len = n_chains
param_names = ['Intercept', 'θ1', 'θ2']

#  ESS
print("\nPER-CHAIN ESS (each chain separately):")
for chain_id, arr in enumerate(post_chains, start=1):
    print(f"\nChain {chain_id}:")
    for j, name in enumerate(param_names):
        ess_j = effective_sample_size(arr[:, j], max_lag=1000)
        n_keep = arr.shape[0]
        frac = ess_j / n_keep
        print(f"{name:10s}  ESS ≈ {ess_j:8.1f}   (ESS / {n_keep} ≈ {frac:.3f})")



