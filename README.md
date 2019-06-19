# SilentMLE

SilentMLE allows for optimal estimation of the silent fraction of some defined synaptic population. It works on electrophysiology data collected using the failure-rate analysis method (FRA; failure rates collected from neurons voltage-clamped at -70mV and +40mV while synapses are electrically stimulated).

SilentMLE works by running a set of detailed, constrained experimental simulations which provide a mapping between ground truth silent values, and biased subsamples of synapses recorded by the traditional failure-rate methods in most experimental setups. Thus, a numerically simulated likelihood function can be constructed which allows for optimal estimation, with significant reductions in both bias and variance of the estimator.

The estimator can work on either raw failure rate data, or on previously estimated fractions using the failure-rate analysis method. It initializes an estimator object where electrophysiologists can specify the parameters of their particular experimental setup, and can provide a release probability distribution if they so choose. The estimator then simulates a likelihood function which should match their particular experiments. Given a set of observations, the estimator returns a joint likelihood function, which allows for maximum-likelihood estimation as well as the specification of confidence intervals and comparison between experimental groups, if so desired.

## Getting Started

### Prerequisites
The main dependencies are: numpy, scipy, matplotlib.

### Initializing the estimator
First, an instance of the estimator object class is initialized. Here, the experimenter can provide experimental details and parameters, and can specify how fine-grained the generated likelihood functions should be.

    estimator = MLESilent(n_simulations = 10000, n_likelihood_points = 20,
      num_trials = 50, failrate_low = 0.2, failrate_high = 0.8)

(More details about possible kwargs can be found in the full class documentation.)

This will run a set of constrained experimental simulations, storing relevant results as class instance attributes. Of interest, all parameters chosen can be viewed in:

    estimator.params #A dictionary of simulation parameters used

### Performing MLE with the estimator
One can perform MLE on either a set of previous FRA estimates, or on pairs of raw failure rates at -70mV and +40mV. Both types are demonstrated below.

    data_estimates = [0.18, 0.27, 0.11, -0.10, -0.02, 0.08]
    joint_likelihood1 = estimator.perform_mle(data_fra, dtype = 'est')

    data_failrates = [[0.2, 0.18], [0.4, 0.36], [0.38, 0.48], [0.58, 0.6], [0.46, 0.31]]
    joint_likelihood2 = estimator.perform_mle(data_failrates, dtype = 'failrate')

By default the joint likelihood function is also plotted. This can be disabled in the method parameters.

## Advanced commands

### Functions to perform experimental simulations

    generate_realistic_constellation() #Runs a set of constrained experimental simulations and returns a subsample of silent/nonsilent synapses each with their own release probability.
    generate_fra_distribution() #Generates a distribution of uncorrected failure-rate estimates from the constrained experimental simulations, starting from a known ground truth of fraction silent synapses.

# Functions to plot main figures from paper

    #Each of these functions will redo all analysis/simulations and will return a fully formatted figure.

    plot_fig1()
    plot_fig1_S1()
    plot_fig1_S2()
    plot_fig2()
    plot_fig4()
    plot_fig4_suppLLR()
