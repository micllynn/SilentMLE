"""
mle.py: Contains main class for running maximum likelihood analysis.

Author: mbfl
Date: 19.9
"""
import numpy as np
import scipy.signal as sp_signal
import matplotlib.pyplot as plt

from .core import *

__all__ = ['Estimator']


class Estimator(object):
    """A max-likelihood estimator object for silent synapse fraction.

    When initialized, generates a set of likelihood functions from
    constrained experimental simulations. The numerically simulated
    likelihood functions, along with observation and hypothesis vectors,
    are stored as attributes in the class instance.

    Parameters
    ----------
    n_simulations : int
        Number of simulated experiments to perform for each hypothesis
    n_likelihood_points : int
        The number of discrete points in the hypothesis-space (values for
        fraction silent synapses) to simulate
    obs_bins  : float
        Bin width for aligning observations to the nearest simulated
        observation
    zeroing : bool
        Denotes whether the observations should be zeroed in the simulation
        (eg negative values artificially set to 0 or not). Simulation and
        experiment should match here.

    **kwargs : experimental constraint parameters. Any of these can be used.
        method : string; can be 'iterative' or 'rand'
            Describes the method for experimental simulation. If
            'iterative' (default), an experimental ensemble of synapses is
            generated through stepwise reduction of some large synapse
            set until a criterion in terms of failure rate is reached.
            If 'rand', an ensemble of synapses is generated mathematically
            through stochastic picking of a small synapse set (i.e. no
            steps and no history involved). 'rand' should provide small
            synapse ensembles that, on average, are faithful subsamples of
            the larger population. 'iterative' imposes some experimental
            bias on the subsamples.
        unitary_reduction : bool
            If method = 'iterative', determines whether synapses are
            eliminated one at a time from the experimental ensembles
            (True), or eliminated in fractional chunks proportional to the
            total n (False).
        frac_reduction : float
            If unitary_reduction is false, then frac_reduction describes
            the fraction of the total n to reduce the recorded population
            by during each step.
        pr_dist_sil : PrDist class instance (see .core for info on PrDist)
            Describes the distribution with which to draw silent synapse
            release probabilities from.
        pr_dist_nonsil : PrDist class instance (see .core for info on PrDist)
            Describes the distribution with which to draw nonsilent synapse
            release probabilities from.
        num_trials : int (default is 50)
            Number of trials to simulate for each voltage-clamp condition
            (e.g.) hyperpolarized and depolarized. Default is 50.
        failrate_low : float (default 0.2)
            The lower bound of hyperpolarized failure rates to accept for the
            synapse ensembles (a theoretical calculation based on release
            probabilities of all synapses - individual experiments may differ
            slightly from this).
        failrate_high : float (default 0.8)
            The upper bound of hyperpolarized failure rates to accept for the
            synapse ensembles (a theoretical calculation based on release
            probabilities of all synapses - individual experiments may differ
            slightly from this).
        n_start : int (default 100)
            The number of synapses to start with for the experimental down-
            sampling. Since ~3-6 nonsilent synapses are typically reached, the
            precise value of n_start does not carry much practical importance.
        graph_ex : boolean (default False)
            Whether to plot an example of the experimental synapse
            downsampling.
        verbose : boolean (default False)
            Toggles verbosity for the simulation on and off for troubleshooting
            or for use during large-scale simulations.
        sim_oversample_factor : int
            Factor by which to oversample experimental simulations by, compared
            to the requested n_simulations. More experimental simulations than
            requested are typically needed to compensate for the fact that some
            simulations simply do not yield usable ensembles of synapses (ie
            failure rates are not in the acceptable bounds.)
                - Note that sim_oversample_factor is only applied to cases with
                silent_fraction > 0.9 (not needed below this).
                - If this function hangs, sim_oversample_factor can be
                increased to compensate.
        sim_oversample_thresh : float
            Threshold of silent_frac after which to start oversampling #sims
            by sim_oversample_factor (to ensure that sufficient numbers of sims
            are returned despite some sims being discarded for failure rate
            ranges not falling in the specified interval).

    Returns
    --------
    self.params : dict
        Stores all params (including those passed by **kwargs)
        in a dictionary.
    self.obs : np.array
        Stores the observation values used to perform the likelihood


    Attributes
    --------
    obs : np.ndarray, shape = (n)
        A vector storing all observations considered for the likelihood
        function.

    hyp : np.ndarray, shape = (m)
        A vector storing all values of the hypothesis (fraction silent)
        for the likelihood function

    fra_dists : np.ndarray, shape = (m)
        An array of arrays storing the FRA estimate distributions for
        each hypothesis found in self.hyp.

    likelihood : np.ndarray, shape = (n, m)
        A vector storing a likelihood for each pair of observations
        (dim1) and hypothesis (dim2)

    params : dict
        A dictionary of experimental and simulation parameters which were used
        to generate self.likelihood.
    """

    def __init__(self, n_simulations=10000, n_likelihood_points=200,
                 obs_bins=0.02, zeroing=False, **kwargs):

        print('\nNew Estimator\n----------------------')

        # Store a dict of args/params in the attribute .params
        self.params = {}
        param_names = ['n_simulations', 'n_likelihood_points', 'obs_bins',
                       'zeroing']
        for ind, param in enumerate([n_simulations, n_likelihood_points,
                                     obs_bins, zeroing]):
            self.params[param_names[ind]] = param
        self.params.update(**kwargs)

        # 1. Generate observations (est. frac. silent) resulting from each
        # hypothesis (frac. silent)
        hyp = np.linspace(0, 0.99, num=n_likelihood_points)
        fra_calc = np.empty(len(hyp), dtype=np.ndarray)

        for ind, silent in enumerate(hyp):
            progress = ind/len(hyp)*100
            print('\r' + f'Generating estimate distributions... {progress:.1f}'
                  + '%', end='')

            # Generate an FRA dist
            fra_calc[ind] = gen_fra_dist(n_simulations=n_simulations,
                                         silent_fraction=silent,
                                         zeroing=zeroing, **kwargs)

        print('\r' + 'Generating estimate distributions... 100%')

        # 2. Create loglikelihood function for each case
        # Bin observations from -200 to 100
        obs = np.arange(-2, 1+2*obs_bins, obs_bins)
        likelihood = np.empty([len(obs), len(hyp)], dtype=np.float64)
        for ind_obs, obs_ in enumerate(obs):
            progress = ind_obs/len(obs)*100
            print('\r' + f'Generating likelihood functions... {progress:.1f}'
                  + '%', end='')

            for ind_hyp, hyp_ in enumerate(hyp):
                # Calculate simulated observations which fall within half an
                # obs_bin of the current obs (ie obs is the closest to them).
                obs_in_range = np.where(
                    np.abs(fra_calc[ind_hyp] - obs_) < obs_bins/2)[0]
                p_obs = len(obs_in_range) / len(fra_calc[ind_hyp])
                likelihood[ind_obs, ind_hyp] = p_obs
        print('\r' + 'Generating likelihood functions... 100%')

        likelihood += 0.0001  # Set likelihoods microscopically away from 0
        # (to avoid log(0) errors)

        self.fra_dists = fra_calc

        self.obs = obs
        self.hyp = hyp
        self.likelihood = likelihood

        print('\n** Estimator initialized **')

    def _smooth_likelihood(self, joint_likelihood,
                           window_length_silfrac=0.05,
                           polyorder=3):
        """Method filters a joint likelihood with a savgol filter using
        intelligently picked params, and returns the result.

        Parameters
        -----------
        joint_likelihood : np.array
            Array expressing the joint likelihood (ie considered across all
            observations.)

        window_length_silfrac : float
            Desired window length for filter, expressed as silent fraction.
            (Converted internally to indices).

        polyorder : int
            Order of polynomial used for smoothing.

        Returns
        ----------
        likelihood_sm : np.array
            Likelihood function smoothed with Savitzky-Golay filter.
        """
        hyp_interval = self.hyp[1] - self.hyp[0]
        window_length_ind = int(round(window_length_silfrac/hyp_interval))

        if np.divmod(window_length_ind, 2)[1] == 0:
            window_length_ind += 1

        likelihood_sm = sp_signal.savgol_filter(
            joint_likelihood,
            window_length=window_length_ind,
            polyorder=polyorder)

        return likelihood_sm

    def estimate(self, data, plot_joint_likelihood=True,
                 dtype='est', use_smoothed=True,
                 smooth_window=0.1, smooth_polyorder=3,
                 return_normed=True, verbose=True):
        """
        Method performs maximum likelihood analysis on a set of data.

        It uses a previous numerically generated likelihood function, and
        returns a joint likelihood distribution of all data. Additionally, it
        can construct a plot of the joint likelihood.

        Parameters
        -------------
        data : array-like
            A 1-d array of data.
            * If dtype = 'est: data must comprise of individual estimates of
            fraction synapses silent, calculated using the failure-rate
            analysis equation and expressed as fractions, not percents
                e.g. data = [0.27, -0.14, 0.63, -0.01]
            * If dtype = 'failrate': data must comprise of pairs of failure
            rate observations in the format (Fh, Fd), expressed as fractions.
                e.g. data = [[0.35, 0.21], [0.56, 0.58], [0.49, 0.27]].

        plot_joint_likehood  : boolean
            Whether to construct a plot of the joint likelihood function.

        dtype : string | 'est' or 'failrate'
            If 'est', then data must be comprised of estimates of silent
            fraction passed through the failure-rate equation.
            If 'failrate', then raw failure rates are given in data.

        use_smoothed : boolean
            Whether to use the smoothed likelihood function (True) or
            the raw likelihood (False).
                * Typically, the smoothed function allows for more
                reliable estimates since it averages out noise
                during the simulations.

        smooth_window : float
            If use_smoothed is True, this specifies the
            desired window length for filter, expressed as silent fraction.
            (Converted internally to indices).

        smooth_polyorder : int
            If use_smoothed is True, this specifies the
            order of polynomial used for smoothing.

        return_normed : boolean
            Whether to return a normalized likelihood or not.

        verbose : boolean
            Whether to print progress statements to the terminal.

        Returns
        --------------
        joint_likelihood : np.ndarray
            The joint likelihood function across all data. Values are indexed
            to self.hyp (hypothesis space, or silent synapse fractions).
        """
        if verbose is True:
            print('\nRunning MLE on data\n----------------------')

        if dtype == 'failrate':
            # Convert failrates to silent fraction estimates and store back
            # in data.
            n_data = len(data)
            data_calc = np.empty(n_data)
            for ind, datum in enumerate(data):
                data_calc[ind] = 1 - np.log(datum[0])/np.log(datum[1])
            data = data_calc

        likelihood_data = np.empty((len(data), len(self.hyp)))
        joint_logl_data = np.zeros((len(self.hyp)), dtype=np.float64)

        for ind, datum in enumerate(data):
            ind_datum_ = np.argmin(np.abs(datum - self.obs))
            likelihood_data[ind] = self.likelihood[ind_datum_, :]
            joint_logl_data += np.log(likelihood_data[ind])

        joint_likelihood = np.exp(joint_logl_data)
        joint_likelihood_norm = joint_likelihood / np.max(joint_likelihood)

        # Large-dataset condition to prevent np.exp({very negative number}):
        _use_logl = np.min(joint_logl_data) < -500

        if use_smoothed == True and _use_logl == False:
            if verbose is True:
                print('Smoothing...')
            joint_likelihood_norm = self._smooth_likelihood(
                joint_likelihood_norm,
                window_length_silfrac=smooth_window,
                polyorder=smooth_polyorder)

        if _use_logl:
            ind_mle_data = np.argmax(joint_logl_data)
        else:
            ind_mle_data = np.argmax(joint_likelihood_norm)
        mle_data = self.hyp[ind_mle_data]

        if plot_joint_likelihood is True:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)

            if _use_logl:
                ax.plot(self.hyp, joint_logl_data)
                ax.set_ylabel('log-likelihood')
            else:
                ax.plot(self.hyp, joint_likelihood_norm)
                ax.set_ylabel('likelihood (norm.)')

            mle_x_ = [self.hyp[ind_mle_data], self.hyp[ind_mle_data]]
            mle_y_ = [ax.get_ylim()[0], ax.get_ylim()[1]]
            ax.plot(mle_x_, mle_y_, '--', color=[0.9, 0.05, 0.15])
            ax.set_xlabel('silent frac.')
            ax.set_xlim([0, 1])
            ax.text(self.hyp[ind_mle_data], 1.1,
                    f'MLE = {mle_data:.2f}', horizontalalignment='left',
                    verticalalignment='bottom', color=[0.9, 0.05, 0.15],
                    fontweight='bold')
            fig.savefig('joint_likelihood.pdf')
            plt.show()

        if verbose is True:
            print(f'MLE = {mle_data*100:.1f}% silent')

        if _use_logl:
            return joint_logl_data
        else:
            if return_normed is True:
                return joint_likelihood_norm
            elif return_normed is False:
                return joint_likelihood
