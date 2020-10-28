# SilentMLE

SilentMLE allows for optimal estimation of the silent fraction of some defined
synaptic population. It works on electrophysiology data collected using the
failure-rate analysis method (FRA; failure rates collected from neurons
voltage-clamped at -70mV and +40mV while synapses are electrically stimulated).

SilentMLE works by running a set of detailed, constrained experimental simulations
which provide a mapping between ground truth silent values, and biased subsamples
of synapses recorded by the traditional failure-rate methods in most experimental
setups. Thus, a numerically simulated likelihood function can be constructed which
allows for optimal estimation, with significant reductions in both bias and
variance of the estimator.

The estimator can work on either raw failure rate data, or on previously estimated
fractions using the failure-rate analysis method. It initializes an estimator
object where electrophysiologists can specify the parameters of their particular
experimental setup, and can provide a release probability distribution if they so
choose. The estimator then simulates a likelihood function which should match
their particular experiments. Given a set of observations, the estimator returns
a joint likelihood function, which allows for maximum-likelihood estimation as
well as the specification of confidence intervals and comparison between
experimental groups, if so desired.

## Getting Started

### Prerequisites
The main dependencies are: python3.6, numpy, scipy, matplotlib.

### Installation on UNIX-like systems (MacOS/Linux)

A setup.py file allows for simple package installation:
```
$ cd {path-to-package}
$ python3 setup.py install
```

## Performing maximum-likelihood estimation

### Initializing the estimator
First, the package is imported and an instance of the `sil.Estimator` class
is initialized. This runs a set of constrained simulations to determine the
mapping between the ground-truth fraction silent in some population, and the
biased measurement returned by electrical stimulation experiments. The
resulting likelihood function is then stored in the estimator
(`estimator.likelihood`), which is then ready to estimate.

```python
import silentmle as sil
estimator = sil.Estimator()
```

Alternately, the `sil.Estimator` class can be initialized with a number of
experimental constraints and/or simulation parameters. A full list of
possible parameters can be found in the full class documentation of
`sil.Estimator`, accessible through `help(sil.Estimator)`. As an example:
```python
estimator = sil.Estimator(n_likelihood_points=500,
      pr_dist_sil=sil.PrDist(sp_stats.uniform),
	  pr_dist_nonsil=sil.PrDist(sp_stats.uniform),
	  num_trials=100, failrate_low=0.4,
      failrate_high=0.6)
```

This simulates using a fine-grained 500 points in the hypothesis-space
(fraction silent); using a uniform release probability distribution for all
synapses; and run the experimental simulations with 100 trials per Vm and an
accepted failure rate range at hyperpolarized values of 0.4<F<0.6.

All simulation and experimental parameters are stored in the class instance
as `estimator.params` for easy access. The observation space and hypothesis
space employed are stored as `estimator.obs` and `estimator.hyp` respectively.
The numerically simulated likelihood function is stored as `estimator.likelihood`.

#### Advanced estimator usage
More advanced control of the Pr distributions is available with the
PrDist class, which is passed as an argument to the Estimator() class:
```python
import scipy.stats as sp_stats

pr_dist = sil.PrDist(dist=sp_stats.gamma,
	args={'a':1, 'scale':1/5.8})
	
estimator = sil.Estimator(pr_dist_sil=pr_dist, pr_dist_nonsil=pr_dist)
```
This creates a distribution object for release probability which is sampled
during the experimental simulations. Any distribution from the scipy.stats
library can be specified, and any keyword arguments should be specified as
a dictionary as above.

### MLE on experimental data
One can perform MLE on either a set of previous FRA estimates, or on pairs
of raw failure rates at -70mV and +40mV. Both types are demonstrated below.

- *For data in FRA estimate form:*
```python
data = [0.18, 0.27, 0.11, -0.10, -0.02, 0.08]
likelihood = estimator.estimate(data, dtype='est')
```

- *For data in raw failure rate form:*
```python
data = [[0.2, 0.18], [0.4, 0.36], [0.38, 0.48], [0.58, 0.6], [0.46, 0.31]]
likelihood = estimator.estimate(data, dtype='failrate')
```

Here, likelihood returns a vector of length `h` containing the joint
likelihood across all observations, where `h` is the hypothesis space
size (i.e. `len(estimator.hyp)`)

## Example of full estimation procedure and output:
In this section, we provide an example code block for the full SilentMLE
estimation, as well as the resulting figure.

It is important to tailor the parameters to one's unique set of
experimental parameters for optimal estimation (see above). However,
this can act as a starting point for the use of the SilentMLE package.

```python
import silentmle as sil
import scipy.stats as sp_stats

estimator = sil.Estimator(n_likelihood_points=500,
      pr_dist_sil=sil.PrDist(sp_stats.uniform),
	  pr_dist_nonsil=sil.PrDist(sp_stats.uniform),
	  num_trials=50, failrate_low=0.2,
      failrate_high=0.8)
	  
# replace data here with fra-calculated silent fraction for each cell
data = [-0.36,  0.37,  0.57,  0.89, -0.43,
        0.49,  0.20,  0.22,  0.16,  0.51,
        0.67,  0.17, -0.48, -0.25,  0.00,
        0.16,  0.16,  0.90,  0.23, -0.15]
likelihood = estimator.estimate(data, dtype='failrate')
```

The resulting figure:
![figure_output] (/ex-fig.pdf)

## Advanced commands

### Core simulation/analysis functions
The following are core simulation and analysis functions. Full
docstrings are available with `help(desired-function)`.

- `sil.core.draw_subsample()`: Experimental simulation drawing a subsample
of synapses from a larger population
- `sil.core.fra(fh, fd)`: Returns FRA estimate of fraction silent, given
a hyperpolarized failure rate (`fh`) and a depolarized failure rate (`fd`).
- `sil.core.gen_fra_dist()`: Generates distribution of FRA values given a 
ground truth silent.
- `sil.core.power_analysis()`: Performs a power analysis (number of samples
required for some specified statistical power) on two distributions
of estimates for fraction synapses silent, which ostensibly come from two
separate populations with distinct true fractions of silent synapses.
  - Utilizes a highly efficient divide-and-conquer algorithm
  (binary search) which has complexity `O(log_{2}(n))`, where n is
  the maximum sample size required.
- `sil.core.power_analysis_llr()`: As above, but performs a power analysis
using log-likelihood ratio testing (hypothesis-based testing) using the
maximum-likelihood estimates.

### Plotting function
Each of these functions will redo all analysis/simulations and will return a
near-fully formatted figure.

- `sil.figures.plot_fig1()`
- `sil.figures.plot_fig2()`
- `sil.figures.plot_fig4()`
- `sil.figures.plot_figS1()`
- `sil.figures.plot_figS2()`
- `sil.figures.plot_figS3()`
- `sil.figures.plot_figS4()`

Note that some figures (eg FigS2-3) are constructed from base figures with
different kwargs.
