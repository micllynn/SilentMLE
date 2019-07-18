# SilentMLE

SilentMLE allows for optimal estimation of the silent fraction of some defined synaptic population. It works on electrophysiology data collected using the failure-rate analysis method (FRA; failure rates collected from neurons voltage-clamped at -70mV and +40mV while synapses are electrically stimulated).

SilentMLE works by running a set of detailed, constrained experimental simulations which provide a mapping between ground truth silent values, and biased subsamples of synapses recorded by the traditional failure-rate methods in most experimental setups. Thus, a numerically simulated likelihood function can be constructed which allows for optimal estimation, with significant reductions in both bias and variance of the estimator.

The estimator can work on either raw failure rate data, or on previously estimated fractions using the failure-rate analysis method. It initializes an estimator object where electrophysiologists can specify the parameters of their particular experimental setup, and can provide a release probability distribution if they so choose. The estimator then simulates a likelihood function which should match their particular experiments. Given a set of observations, the estimator returns a joint likelihood function, which allows for maximum-likelihood estimation as well as the specification of confidence intervals and comparison between experimental groups, if so desired.

## Getting Started

### Prerequisites
The main dependencies are: numpy, scipy, matplotlib.

### Installation on UNIX-like systems (MacOS/Linux)

A setup.py file allows for simple package installation:
```
$ cd ~/Downloads/SilentMLE*
$ python setup.py install
```

### Initializing the estimator
First, the package is imported and an instance of the `sil.Estimator` class is initialized. This runs a set of constrained simulations to determine the mapping between the ground-truth fraction silent in some population, and the biased measurement returned by electrical stimulation experiments. The resulting likelihood function is then stored in the estimator (`estimator.likelihood`), which is then ready to estimate.

```python
import silentmle as sil
estimator = sil.Estimator()
```

Alternately, the `sil.Estimator` class can be initialized with a number of experimental constraints and/or simulation parameters:
```python
#Simulate using a fine-grained 500 points in the hypothesis-space (fraction silent);
#use a uniform release probability distribution for the synapses; and
#run the experimental simulations with 100 trials per Vm and an accepted failure
#rate range at hyperpolarized values of 0.4<F<0.6.
estimator = sil.Estimator(n_likelihood_points = 500,
      pr_dist = 'uniform', num_trials = 100, failrate_low = 0.4,
      failrate_high = 0.6)
```

(More details about possible kwargs can be found in the full class documentation of `sil.Estimator`)

All simulation and experimental parameters are stored in the class instance for easy access:
```python
estimator.params #A dictionary of simulation parameters used
```

### Performing maximum likelihood estimation on a set of data
One can perform MLE on either a set of previous FRA estimates, or on pairs of raw failure rates at -70mV and +40mV. Both types are demonstrated below.

```python
#For data in FRA estimate form
data = [0.18, 0.27, 0.11, -0.10, -0.02, 0.08]
likelihood = estimator.estimate(data, dtype = 'est')

#For data in raw failure rate form
data = [[0.2, 0.18], [0.4, 0.36], [0.38, 0.48], [0.58, 0.6], [0.46, 0.31]]
likelihood = estimator.estimate(data, dtype = 'failrate')
```

Here, likelihood returns a vector of length `h` containing the joint likelihood across all observations, where `h` is the hypothesis space size (i.e. `len(estimator.hyp)`)

## Advanced commands

### Core functions to perform experimental simulations and FRA analysis

```python
sil.core.draw_subsample() #Runs a set of constrained experimental simulations and returns a subsample of silent/nonsilent synapses each with their own release probability.

sil.core.fra(fh, fd) #Estimates, using the FRA equation, the fraction silent synapses given failure rates at hyperpolarized (fh) and depolarized (fd) membrane potentials.

sil.core.gen_fra_dist() #Generates a distribution of uncorrected failure-rate estimates from the constrained experimental simulations, starting from a known ground truth of fraction silent synapses.
```

# Functions to plot main figures from paper

```python
#Each of these functions will redo all analysis/simulations and will return a fully formatted figure.

sil.figures.plot_fig1()
sil.figures.plot_fig1_S1()
sil.figures.plot_fig1_S2()
sil.figures.plot_fig2()
sil.figures.plot_fig4()
sil.figures.plot_fig4_suppLLR()

```
