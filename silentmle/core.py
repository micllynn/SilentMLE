"""
core.py: Main functions for simulating synapse reductions, generating
synthetic FRA distributions, and performing power analysis.

Author: mbfl
Date: 19.9
"""

import numpy as np
import scipy as sp
import scipy.stats as sp_stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


__all__ = ['binomial_fill', 'PrDist', 'draw_subsample', 'fra', 'gen_fra_dist',
           'power_analysis', 'power_analysis_llr']


def binomial_fill(n_slots, frac_silent, n_sims=1):
    """
    Function probabilistically fills n_slots with silent synapses based on a
    binomial distribution given frac_silent. n_sims total simulations are run
    and returned in a vector silent_draw.

    Parameters
    ----------
    n_slots : int
        Number of slots to fill.
    frac_silent : float
        Fraction of silent synapses, 0<frac_silent<1.
    n_sims : int (default 1)
        Number of simulations to return

    Returns
    ----------
    silent_draw : np.ndarray
        Vector of draws, with shape n_sims. Each element is an integer
        denoting the number of silent synapses, out of n_slots,
        for that particular simulation.
    """
    n_silent = np.arange(0, n_slots + 1)
    prob_n_silent = np.empty(n_slots + 1)

    for ind, synapse in enumerate(n_silent):
        num_combinations = sp.special.comb(n_slots, n_silent[ind])
        prob_n_silent[ind] = ((frac_silent) ** n_silent[ind]) \
            * ((1 - frac_silent) ** (n_slots - n_silent[ind])) \
            * num_combinations

    silent_draw = np.random.choice(n_silent, size=n_sims, p=prob_n_silent)

    return silent_draw


class PrDist(object):
    """
    Class holds information about a scipy.stats probability dist.
    (Passed to draw_subsample to specify the Pr distribution for
    a group of synapses.)

    Parameters
    ---------
    dist : scipy.stats distribution
        The distribution to draw from.
    args: dict (default {})
        A dictionary holding kwargs for the scipy.stats distribution.
        (e.g. args={'shape': 1, 'scale': 2} will get passed as **args
        to dist when it is initialized.)
        * Defaults to an empty dictionary, i.e. default args for the dist.
    """

    def __init__(self, dist, args={}):
        self.dist = dist(**args)

    def __call__(self, size):
        """
        Draws an array of random values from the specified distribution,
        with a certain size.

        Parameters
        ----------
        size : tuple
            Size of the array to draw (eg size=(1, 3))

        Returns
        rvs : np.ndarray
            Numpy array of random values with rvs.shape=size
        """
        return self.dist.rvs(size=size)


def draw_subsample(silent_fraction=0.5, n_simulations=100,
                   method='iterative',
                   pr_dist_sil=PrDist(sp_stats.uniform),
                   pr_dist_nonsil=PrDist(sp_stats.uniform),
                   n_start=100,
                   failrate_low=0.2, failrate_high=0.8,
                   plot_ex=False, unitary_reduction=True,
                   frac_reduction=0.2, sim_oversample_factor=4,
                   verbose=True):
    """
    Function which simulates a set of minimum electrical stimulation experi-
    ments and generates a subset of silent and nonsilent synapses sampled
    from some larger population with defined parameters.

    For each simulation, the algorithm does the following:
        1. Starts with n_start synapses. A binomial distribution is used to
            assign some fraction of these synapses as silent, with
            silent_fraction acting as p(silent).
            Each synapse is assigned an independent release probability from
            the distribution defined in pr_dist.
        2. We then simulate the gradual reduction of electrical stimulation
            intensity to pick some subset of the total synapse population which
            has a hyperpolarized failure rate in some acceptable range
            (e.g. 20%-80% by default).
            A. If unitary_reduction is True, then synapses are eliminated
                stochastically from the recorded ensemble one at a time until
                the theoretical failure rate of the population at
                hyperpolarized potentials (F = Pr(1)*Pr(2)*...*Pr(i)
                for synapses n = 1, 2, ... i.) reaches the imposed bounds.
            B. If unitary_reduction is False, then fractions of the total
                recorded synapse population are removed each round (according
                to frac_reduction) until the theorietical failure rate of the
                population at hyperpolarized potentials (as defined above)
                reaches the imposed bounds.
                    - Yields a subset of synapses which is a biased
                    representation of the larger set (in terms of both p_release
                    frac_silent). It reflects experimental constraints and
                    is used for the main results of the paper.
        3. Immediately upon reaching this failure-rate criterion, the set of
        synapses which remain for this experiment are stored in the
        appropriate output parameters and the simulation is considered
        'completed'. The output parameters can then be used to run a failure-
        rate analysis experiment in subsequent functions (e.g.
        gen_fra_dist(). )


    Parameters
    ---------
    silent_fraction : float
        the fraction of silent synapses in the total population
    n_simulations : int
        The total number of simulated synapse ensembles to create
    method : string;  'iterative' or 'rand'
        Describes the method for experimental simulation. If 'iterative'
        (default), an experimental ensemble of synapses is generated through
        stepwise reduction of some large synapse set until a criterion in
        terms of failure rate is reached. If 'rand', an ensemble of synapses
        is generated mathematically through stochastic picking of a small
        synapse set (i.e. no steps and no history involved). 'rand' should
        provide small synapse ensembles that, on average, are faithful
        subsamples of the larger population. 'iterative' imposes some
        experimental bias on the subsamples.
    pr_dist_sil : PrDist class instance
        Describes the distribution with which to draw silent synapse
        release probabilities from.
    pr_dist_nonsil : PrDist class instance
        Describes the distribution with which to draw nonsilent synapse
        release probabilities from.
    n_start : int (default 100)
        The number of synapses to start with for the experimental down-
        sampling. Since ~3-6 nonsilent synapses are typically reached, the
        precise value of n_start does not carry much practical importance.
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
    plot_ex : boolean (default False)
        Whether to plot an example of the experimental synapse downsampling.
    unitary_reduction : bool
        If method = 'iterative', determines whether synapses are eliminated
        one at a time from the experimental ensembles (True), or are
        eliminated in fractional chunks proportional to the total n (False).
    frac_reduction : float
        If unitary_reduction is false, then frac_reduction describes the
        fraction of the total n to reduce the recorded population by
        each step.
    sim_oversample_factor : int
        Factor by which to oversample experimental simulations by, compared
        to the requested n_simulations. More experimental simulations than
        requested are typically needed to compensate for the fact that some
        simulations simply do not yield usable ensembles of synapses (ie their
        failure rates are not in the acceptable bounds.)
            - Note that sim_oversample_factor is only applied to cases with
            silent_fraction > 0.9 (not needed below this).
            - If this function hangs, sim_oversample_factor can be increased
            to compensate.
    verbose : boolean (default False)
        Toggles verbosity for the simulation on and off for troubleshooting
        or for use during large-scale simulations.

    Returns
    -----------
    nonsilent_syn_group : np.ndarray
        An array of length n where array[n] contains the number of nonsilent
        synapses in simulation n
    silent_syn_group : np.ndarray
        An array of length n where array[n] contains the number of silent
        synapses in simulation n
    pr_nonsilent_syn_group : np.ndarray
        An array of length n where array[n] contains the release probabilities
        of all nonsilent synapses in simulation n
    pr_silent_syn_group : np.ndarray
        An array of length n where array[n] contains the release probabilities
        of all silent synapses in simulation n

    """

    ###########
    # Depending on subsampling method, either reduce iteratively by fractions,
    # or simply pick random synapses
    if method == 'iterative':
        # Oversample simulations needed due to the fact that some sims will not
        # have sufficient Pr's and n's to produce desired failure rate
        if silent_fraction >= 0.9:
            n_sims_oversampled = n_simulations * sim_oversample_factor
        elif silent_fraction < 0.9:
            n_sims_oversampled = n_simulations * 2

        # Generate an initial large constellation of synapses
        # (n_start in size) drawm from a binomial filling procedure
        silent_syn_group = binomial_fill(n_start, silent_fraction,
                                         n_sims=n_sims_oversampled)
        nonsilent_syn_group = (n_start
                               * np.ones_like(silent_syn_group)) \
                               - silent_syn_group

        pr_silent_tomask = np.ones((n_sims_oversampled, n_start),
                                   dtype=np.bool)
        pr_nonsilent_tomask = np.ones_like(pr_silent_tomask)

        # Release prob: Generate mask for np.ma.array type where only
        # existing synapses are unmasked
        for sim in range(n_sims_oversampled):
            pr_silent_tomask[sim, 0:silent_syn_group[sim]] = False
            pr_nonsilent_tomask[sim, 0:nonsilent_syn_group[sim]] = False

        # Release prob: Generate release probability distributions drawn
        # from a random distribution
        random_pr_array_sil = pr_dist_sil(size=(n_sims_oversampled,
                                                n_start))
        random_pr_array_nonsil = pr_dist_nonsil(size=(n_sims_oversampled,
                                                      n_start))

        pr_silent_syn_group = np.ma.array(random_pr_array_sil,
                                          mask=pr_silent_tomask)
        pr_nonsilent_syn_group = np.ma.array(random_pr_array_nonsil,
                                             mask=pr_nonsilent_tomask)

        # Store whether each sim has reached its target in reached_target
        reached_target = np.zeros(n_sims_oversampled, dtype=np.bool)
        prev_sum_reached_target = 0
        sum_reached_target = 0

        # Counts number of reductions for frac_reduction
        frac_reduction_count_ = 0
        # Stores target val for all reductions
        frac_reduction_target_ = frac_reduction * n_start

        # --------------------
        # Reduce synapses in group for each simulation until
        # all have reached target
        # -------------------
        while np.sum(reached_target) < n_simulations:
            # Precalculate probability of removing silent synapse
            p_remove_silent = silent_syn_group \
                / (silent_syn_group + nonsilent_syn_group)

            # Case where single synapses are removed:
            sil_syn_to_remove = np.ma.array(np.random.rand(n_sims_oversampled)
                                            < p_remove_silent)

            # Remove constellations which have reached their target from the
            # list of synapses to remove
            sil_syn_to_remove[np.where(reached_target == True)[0]] = np.ma.masked

            # Remove synapse and correct for those that go below 0
            ind_remove_silent = np.ma.where(sil_syn_to_remove == 1)[0]
            ind_remove_silent[ind_remove_silent < 0] = 0
            silent_syn_group[ind_remove_silent] -= 1
            silent_syn_group[silent_syn_group < 0] = 0  # Remove below 0

            ind_remove_nonsilent = np.ma.where(sil_syn_to_remove == 0)[0]
            ind_remove_nonsilent[ind_remove_nonsilent < 0] = 0
            nonsilent_syn_group[ind_remove_nonsilent] -= 1
            nonsilent_syn_group[nonsilent_syn_group < 0] = 0  # remove below 0

            # Update release probabilities
            pr_silent_syn_group[ind_remove_silent,
                                silent_syn_group[ind_remove_silent]] \
                                = np.ma.masked
            pr_nonsilent_syn_group[ind_remove_nonsilent,
                                   nonsilent_syn_group[ind_remove_nonsilent]] \
                                   = np.ma.masked

            # Check how many sims have reached their target failrate.
            if unitary_reduction is True:
                # Condition where each iteration is checked.

                # Modify vector denoting those that have reached target,
                # calculate sum of all reached, and display it
                failrate_thisiteration = np.ma.prod(1 - pr_nonsilent_syn_group,
                                                    axis=1)
                reached_target_inds = np.logical_and(failrate_thisiteration
                                                     < failrate_high,
                                                     failrate_thisiteration
                                                     > failrate_low)

                reached_target[reached_target_inds] = True

                prev_sum_reached_target = sum_reached_target
                sum_reached_target = np.sum(reached_target)

            elif unitary_reduction is False:
                # Condition where multiple iters must pass
                frac_reduction_count_ += 1  # Update the count of n's reduced

                if frac_reduction_count_ >= frac_reduction_target_:
                    # Modify vector denoting those who have reached target
                    failrate_thisiteration = np.ma.prod(
                        1 - pr_nonsilent_syn_group, axis=1)
                    reached_target_inds = np.logical_and(failrate_thisiteration
                                                         < failrate_high,
                                                         failrate_thisiteration
                                                         > failrate_low)

                    reached_target[reached_target_inds] = True

                    prev_sum_reached_target = sum_reached_target
                    sum_reached_target = np.sum(reached_target)

                    # Recalculate a new target and set count to 0
                    frac_reduction_target_ = np.ceil(
                        frac_reduction * np.mean(silent_syn_group
                                                 + nonsilent_syn_group))
                    frac_reduction_count_ = 0

            # Print an in-place update about % progress
            if prev_sum_reached_target is not sum_reached_target \
               and sum_reached_target > 0 and verbose is True:
                to_print = '\r' + '\tRunning subgroup reduction... ' + str(
                        sum_reached_target * 100 / n_simulations) + '%'
                print(to_print, end='')

        if verbose is True:
            print('\r\tRunning subgroup reduction... 100.00% ')

        # Filter out successes
        nonsilent_syn_group \
            = nonsilent_syn_group[reached_target][0:n_simulations]
        silent_syn_group \
            = silent_syn_group[reached_target][0:n_simulations]
        pr_silent_syn_group \
            = pr_silent_syn_group[reached_target][0:n_simulations, :]
        pr_nonsilent_syn_group \
            = pr_nonsilent_syn_group[reached_target][0:n_simulations, :]

    elif method == 'rand':
        n_sims_os = n_simulations
        # Draw random assortments
        slots = np.arange(1, 101)

        nonsilent_syn_group = []
        silent_syn_group = []
        pr_silent_syn_group = np.array([]).reshape(0, slots[-1])
        pr_nonsilent_syn_group = np.array([]).reshape(0, slots[-1])

        for slot in slots:
            # indices of silent/nonsilent synapses to take
            n_silent = binomial_fill(slot, silent_fraction,
                                     n_sims=n_sims_os)

            random_pr_array_sil = pr_dist_sil(size=(n_sims_os, slots[-1]))
            random_pr_array_nonsil = pr_dist_nonsil(size=(n_sims_os, slots[-1]))

            pr_silent = random_pr_array_sil
            pr_nonsilent = random_pr_array_nonsil

            pr_silent_tomask = np.ones_like(pr_silent, dtype=np.bool)
            pr_nonsilent_tomask = np.ones_like(pr_silent, dtype=np.bool)

            # Release prob: Generate mask for np.ma.array type
            # where only existing synapses are unmasked
            for sim in range(n_sims_os):
                pr_silent_tomask[sim, 0:n_silent[sim]] = False
                pr_nonsilent_tomask[sim, 0:(slot
                                            - n_silent[sim])] = False

            # Release prob: Generate release probability distributions
            # drawn from a random distribution
            pr_silent = np.ma.array(pr_silent,
                                    mask=pr_silent_tomask)
            pr_nonsilent = np.ma.array(pr_nonsilent,
                                       mask=pr_nonsilent_tomask)

            # failrate:
            fails_nonsilent = np.ma.prod(1 - pr_nonsilent, axis=1).data
            criterion = np.logical_and(fails_nonsilent > failrate_low,
                                       fails_nonsilent < failrate_high)

            # keep only simulations which reach criterion
            nonsilent_syn_group = np.append(nonsilent_syn_group,
                                            pr_nonsilent[criterion, :]
                                            .count(axis=1))
            silent_syn_group = np.append(silent_syn_group,
                                         pr_silent[criterion, :]
                                         .count(axis=1))
            pr_silent_syn_group = np.ma.append(pr_silent_syn_group,
                                               pr_silent[criterion, :],
                                               axis=0)
            pr_nonsilent_syn_group = np.ma.append(pr_nonsilent_syn_group,
                                                  pr_nonsilent[criterion, :],
                                                  axis=0)

            to_print = '\r' + '\tRunning randomized selection... ' + str(
                    slot * 100 / slots[-1]) + '%'
            print(to_print, end='')

        # Filter out n_sims only
        sims_keep = np.random.choice(
            np.arange(len(nonsilent_syn_group)), size=n_simulations,
            replace=False)

        nonsilent_syn_group = nonsilent_syn_group[sims_keep].astype(np.int)
        silent_syn_group = silent_syn_group[sims_keep].astype(np.int)
        pr_silent_syn_group = pr_silent_syn_group[sims_keep, :]
        pr_nonsilent_syn_group = pr_nonsilent_syn_group[sims_keep, :]

    # -------------------
    # Create visual depiction of synapses
    # ---------
    if plot_ex is True:
        # By default only show 20 sims
        visual_synapses = np.zeros((100, np.max(
            nonsilent_syn_group + silent_syn_group)))

        for sim in range(100):
            visual_synapses[sim, 0:nonsilent_syn_group[sim]] = 255
            visual_synapses[sim, nonsilent_syn_group[sim]:
                            nonsilent_syn_group[sim]
                            + silent_syn_group[sim]] = 128

        fig = plt.figure(figsize=(4, 10))
        ax = fig.add_subplot(1, 1, 1)

        ax.imshow(visual_synapses, cmap='binary')
        ax.set_xlabel('Synapses')
        ax.set_ylabel('Simulations')
        plt.show()

    return nonsilent_syn_group, silent_syn_group, \
        pr_nonsilent_syn_group, pr_silent_syn_group


def fra(fh, fd):
    """
    Convenience function which returns fraction silent given the failure rate
    at hyperpolarized and depolarized potentials.

    Parameters
    ---------
    fh : float, 0<fh<1
        Experimental failure rate at hyperpolarized values
    fd : float, 0<fd<1
        Experimental failure rate at depolarized Values

    Returns
    -------
    fra_est : float
        An estimate of fraction silent synapses in the synapse population.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        fra_est = 1 - np.log(fh) / np.log(fd)

    return fra_est


def gen_fra_dist(silent_fraction, method='iterative',
                 pr_dist_sil=PrDist(sp_stats.uniform),
                 pr_dist_nonsil=PrDist(sp_stats.uniform),
                 num_trials=100, n_simulations=10000, n_start=100,
                 zeroing=False, graph_ex=False, verbose=False,
                 unitary_reduction=False, frac_reduction=0.2,
                 binary_vals=False, failrate_low=0.2,
                 failrate_high=0.8,
                 sim_oversample_factor=8):
    """
    This function generates a distribution of estimates for fraction silent
    synapses returned by the failure-rate analysis estimator, given a single
    true value for fraction silent.

    First, a set of experimentally realistic synapse subsets are produced in
    draw_subsample. Then, a stochastic failure-rate
    experiment is performed for n_simulation of these subsets. Finally, the
    results are stored in fra_calc.

    Parameters
    --------
    binary_vals : boolean
        Specifies whether the function should draw experimental subsets and
        perform simulated FRA experiments on them to construct a set of
        estimates for fraction silent synapses in fra_calc
        (binary_vals = False), or whether the function should simply return
        a set of binary values (0 : not silent; 1 : silent) which could
        correspond to single-synapse experiments (glutamate uncaging or
        minimum electrical stimulation experiments).

    **Parameters which are passed to the draw_subsample fn:
        silent_fraction : float
            the fraction of silent synapses in the total population
        n_simulations : int
            The total number of simulated synapse ensembles to create
        method : string;  'iterative' or 'rand'
            Describes the method for experimental simulation. If 'iterative'
            (default), an experimental ensemble of synapses is generated through
            stepwise reduction of some large synapse set until a criterion in
            terms of failure rate is reached. If 'rand', an ensemble of synapses
            is generated mathematically through stochastic picking of a small
            synapse set (i.e. no steps and no history involved). 'rand' should
            provide small synapse ensembles that, on average, are faithful
            subsamples of the larger population. 'iterative' imposes some
            experimental bias on the subsamples.
        unitary_reduction : bool
            If method = 'iterative', determines whether synapses are eliminated
            one at a time from the experimental ensembles (true), or are
            eliminated in fractional chunks proportional to the total n (false).
        frac_reduction : float
            If unitary_reduction is false, then frac_reduction describes the
            fraction of the total n to reduce the recorded population by
            each step.
        pr_dist_sil : PrDist class instance
            Describes the distribution with which to draw silent synapse
            release probabilities from.
        pr_dist_nonsil : PrDist class instance
            Describes the distribution with which to draw nonsilent synapse
            release probabilities from.
        num_trials : int (default  50)
            Number of trials to simulate for each voltage-clamp condition (e.g.)
            hyperpolarized and depolarized.
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
        plot_ex : boolean (default False)
            Whether to plot an example of the experimental synapse downsampling.
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
                increased to compensate


    Returns
    -------
    fra_calc : np.ndarray
        A vector of length n_simulations which contains independent
        estimates of fraction silent synapses for some simulated population
        of synapses where silent_fraction are known to be silent.
    """

    if verbose is True:
        print('Generating FRA distribution with p(silent) = ',
              str(silent_fraction), ':')

    if binary_vals is True:
        fra_calc = np.zeros(n_simulations)
        fra_calc[0:int(silent_fraction * n_simulations)] = 1

        return fra_calc

    # First, generate realistic groups of neurons
    nonsilent_syn_group, silent_syn_group, \
        pr_nonsilent_syn_group, pr_silent_syn_group \
        = draw_subsample(method=method,
                         pr_dist_sil=pr_dist_sil,
                         pr_dist_nonsil=pr_dist_nonsil,
                         n_simulations=n_simulations,
                         n_start=n_start,
                         silent_fraction=silent_fraction,
                         failrate_low=failrate_low,
                         failrate_high=failrate_high,
                         unitary_reduction=unitary_reduction,
                         frac_reduction=frac_reduction,
                         sim_oversample_factor=sim_oversample_factor,
                         verbose=verbose)

    # Calculate p(failure) mathematically for hyperpol,
    # depol based on product of (1- pr)
    math_failure_rate_hyperpol = np.ma.prod(1 - pr_nonsilent_syn_group,
                                            axis=1).compressed()

    pr_all_depol = np.ma.append(pr_nonsilent_syn_group,
                                pr_silent_syn_group, axis=1)
    math_failure_rate_depol = np.ma.prod(1 - pr_all_depol,
                                         axis=1).compressed()

    # Simulate trials where failure rate is binary, calculate fraction fails
    sim_failure_rate_hyperpol = np.sum(
        np.random.rand(n_simulations, num_trials) < np.tile(
            math_failure_rate_hyperpol,
            (num_trials, 1)).transpose(), axis=1) / num_trials

    sim_failure_rate_depol = np.sum(
        np.random.rand(n_simulations, num_trials) < np.tile(
            math_failure_rate_depol,
            (num_trials, 1)).transpose(), axis=1) / num_trials

    # Calculate failure rate
    fra_calc = fra(sim_failure_rate_hyperpol,
                   sim_failure_rate_depol)

    # Filter out oddities
    fra_calc[fra_calc == -(np.inf)] = 0
    fra_calc[fra_calc == np.inf] = 0
    fra_calc[fra_calc == np.nan] = 0
    fra_calc = np.nan_to_num(fra_calc)

    if zeroing is True:
        fra_calc[fra_calc < 0] = 0

    return fra_calc


def power_analysis(fra_dist_1, fra_dist_2, init_guess=2048,
                   sample_draws=2000,
                   alpha=0.05, beta=0.1, stat_test='ranksum',
                   ctrl_n=False, verbosity=True):
    """
    Performs a power analysis on two distributions of estimates for fraction
    synapses silent, which ostensibly come from two separate populations with
    distinct true fractions of silent synapses.

    Power analysis is performed numerically through typical Monte Carlo methods.
    The search for the minimum sample size which yields statistical
    distinguishability between the population is a computationally
    efficient divide-and-conquer algorithm (binary search). It relies on the
    assumption that if the true minimum sample size yielding statistical
    significance is min_sample, then all sample < min_sample will be
    nonsigificant and all sample > min_sample will be significant. This yields
    a simple algorithm which converges in the true solution quickly and has
    has complexity O(log_{2}(n)) where n is the maximum sample size considered.

    Algorithm:
    1. Starts at init_guess sample size to compare between populations. Call
        this guess_(n) (at step n of the simulation) from now on.
        A. Draw sample_draws paired comparisons, each consisting of init_guess
            independent samples, from the two distributions for an initial
            comparison.
        B. For each of sample_draws pairs, a statistical test according to
            stat_test is performed to compare them and the results for all
            sample_draws comparisons are notated as significant (p_val < alpha)
            or not (p_val > alpha).
        C. Since we know the two populations do indeed result from distinct
            ground truth silent fraction values, we can infer that any
            comparisons which do not show significance are false. Therefore, we
            calculate a rate of false negatives and compare to beta.
        D. If p(false negative | guess_(n)) < beta, then the sample size is
            deemed to be sufficient to statistically distinguish these two
            populations and is marked as such. Else: The sample size is deemed
            to be insufficient.
    2. If the sample size (guess_) is sufficient to statistically distinguish
        the popuations, then the next sample size that is checked is
        guess_(n+1) = guess_(n) - guess_(n)/2. Else: The samples are deemed to
        be indistinguishable (i.e. too large of a sample is needed to
        distinguish them.) This prevents needlessly large sample sizes from
        being considered.
        A. Perform the same set of sample_draws comparisons and beta calculation
            as described above in (1)
    3. If the sample size (guess_) is sufficient to statistically distinguish
        the popuations, then the next sample size that is checked goes down by
        half of the difference between the last and current guess:
            guess_(n+1) = guess_(n) - [abs(guess_(n) - guess_(n-1))] / 2
        If guess_ was not sufficient to statistically distinguish the
        poplations, then the next sample size that is checked goes up by
        half of the difference between the last and current guess:
            guess_(n+1) = guess_(n) + [abs(guess_(n) - guess_(n-1))] / 2
        A. Perform the same set of sample_draws comparisons and beta calculation
            as described above in (1)
    4. Repeat 3 as needed until abs(guess_(n) - guess(n-1)) = 1. Then the true
    minimum sample size is whichever of {guess_(n), guess_(n-1)} is significant.

    Parameters
    ---------------
    fra_dist_1 : np.ndarray | .shape = (n)
        Vector of returned FRA values from population A (from generate_fra_
        distribution)
    fra_dist_2 : np.ndarray | .shape = (n)
        Vector of returned FRA values from population B (from generate_fra_
        distribution)
    init_guess : int
        An initial guess for number of samples required to discriminate the
        two populations. The power analysis starts its search here. It is useful
        to start at intervals of 2^n. (Default 2048)
    sample_draws : int
        How many paired samples to draw from both fra_dist_1 and fra_dist_2 to
        run paired statistical comparisons on during the Monte Carlo simulation.
    alpha : float |  0 < alpha < 1
        Alpha value for the power analysis (type 1 errors, false positives).
    beta : float |  0 < beta < 1
        Beta value for the power analysis (type 2 errors, false negatives)
    stat_test : string | 'ranksum' or 'chisquare'
        Specifies the type of statistical test to use to compare the populations
        of failure rate values.
    ctrl_n : False or int
        If false, for paired comparisons both groups have the same sample size
        drawn from them. If integer, then one group has a fixed sample size
        (a control n, typically far less than the experimental group's n) while
        the other group has a sample size which varies as above throughout
        the algorithm until the
    verbosity : bool
        Toggles verbose reporting of the estimated number of samples as the
        Monte Carlo simulation proceeds.

    Returns
    ------------
    min_samplesize_required : int
        The minimum number of samples required to reach the given alpha and
        beta levels of statistical confidence
    """

    # Initialize variables
    min_sample_reached = False
    current_guess = init_guess
    last_guess = 0
    last_correct = init_guess  # The last correct guess made.
    final_iter = False

    # Iterate through reductions in sample size
    while min_sample_reached is False:
        if verbosity is True:
            print('\t\tn_samples: ', str(current_guess), ' ... ', end='')

        if abs(current_guess - last_guess) < 1.5:
            final_iter = True

        # Pick indices for each sample set
        if ctrl_n is False:
            inds_fra1_ = np.random.randint(0, len(fra_dist_1),
                                           size=(sample_draws, current_guess))
        else:
            inds_fra1_ = np.random.randint(0, len(fra_dist_1),
                                           size=(sample_draws, ctrl_n))

        inds_fra2_ = np.random.randint(0, len(fra_dist_2),
                                       size=(sample_draws, current_guess))
        pvals_ = np.empty([sample_draws])

        # For each sample set, perform rank-sum test and store in pvals_temp
        for ind_power in range(inds_fra1_.shape[0]):
            if stat_test == 'ranksum':
                stat, pval = sp_stats.ranksums(
                    fra_dist_1[inds_fra1_[ind_power, :]],
                    fra_dist_2[inds_fra2_[ind_power, :]])
            elif stat_test == 'chisquare':
                obs1_silent_counts = int(np.sum(
                    fra_dist_1[inds_fra1_[ind_power, :]]))
                obs1_nonsilent_counts = int(current_guess - obs1_silent_counts)
                obs2_silent_counts = int(np.sum(
                    fra_dist_2[inds_fra2_[ind_power, :]]))
                obs2_nonsilent_counts = int(current_guess - obs2_silent_counts)

                contingency = np.array(
                    [[obs1_silent_counts, obs1_nonsilent_counts],
                     [obs2_silent_counts, obs2_nonsilent_counts]])
                if obs1_nonsilent_counts + obs2_nonsilent_counts == 0:
                    pval = 1
                elif obs1_silent_counts + obs2_silent_counts == 0:
                    pval = 1
                else:
                    chi2, pval, dof, expected = sp_stats.chi2_contingency(
                        contingency, correction=False)

            pvals_[ind_power] = pval

        # Calc fraction of draws where pvals_ reaches alpha significance
        fraction_alpha_signif = np.sum(pvals_ < alpha)

        # If beta significance not reached
        if (fraction_alpha_signif / len(pvals_)) > (1 - beta):
            if verbosity is True:
                print('*')

            last_guess_next = current_guess
            current_guess_next = round(
                current_guess - (abs(current_guess - last_guess) / 2))

            last_correct = current_guess

            last_guess = last_guess_next
            current_guess = current_guess_next

            if final_iter is True:
                min_sample_reached = True
                min_samplesize_required = last_correct
                print('\t\t\tn_final: ', str(last_correct))

                break

        # If beta significance reached
        elif (fraction_alpha_signif / len(pvals_)) < (1 - beta):
            if verbosity is True:
                print('NS')

            # Catch for beta significance being reached on the first guess
            if current_guess == init_guess:
                min_sample_reached = True
                min_samplesize_required = current_guess
                break

            last_guess_next = current_guess
            current_guess_next = round(
                current_guess + (abs(current_guess - last_guess) / 2))

            last_guess = last_guess_next
            current_guess = current_guess_next

            if final_iter is True:
                min_sample_reached = True
                min_samplesize_required = last_correct
                print('\t\t\tn_final: ', str(last_correct))

                break

    return min_samplesize_required


def power_analysis_llr(fra_dist_1, likelihoods,
                       ind_null_silent_truefrac=0,
                       init_guess=2048,
                       sample_draws=2000,
                       alpha=0.05, beta=0.2,
                       verbosity=True,
                       plot_ex=False):
    """
    Performs a power analysis using loglikelihood ratio testing (Wilks' theorem)
    on a distribution of estimated silent synapse fractions generated from a
    single ground truth. Requires a numerically simulated likelihood function
    to compare to null hypothesis (i.e. fraction silent synapses = 0)

    Power analysis is performed numerically through typical Monte Carle methods.
    The search for the minimum sample size which yields statistical
    distinguishability between the population is a computationally
    efficient divide-and-conquer algorithm (binary search). It relies on the
    assumption that if the true minimum sample size yielding statistical
    significance is min_sample, then all sample < min_sample will be
    nonsigificant and all sample > min_sample will be significant. This yields
    a simple algorithm which converges in the true solution quickly and has
    has complexity O(log_{2}(n)) where n is the maximum sample size considered.


    Algorithm:
    1. Starts at init_guess sample size to compare between populations. Call
        this guess_(n) (at step n of the simulation) from now on.
        A. Draw sample_draws experiments, each consisting of init_gues
            independent samples, from the fra distribution.
        B. For each of sample_draws experiments, compute the joint loglikelihood
            function for all observations. Compute a loglikelihood ratio
            between 1) MLE of the joint loglikelihood function and 2) the null
            hypothesis where fraction silent synapses = 0:
                D = -2[ln(likelihood(ind_MLE)) - ln(likelihood(ind_null))]
                (D ~ chi2(df=1))
            Assess p_val by comparing to a chi2 distribution. If p_val < alpha,
            the comparison is significant.

        C. Since we know the two populations do indeed result from distinct
            ground truth silent fraction values, we can infer that any
            comparisons which do not show significance are false. Therefore, we
            calculate a rate of false negatives and compare to bseta.
        D. If p(false negative | guess_(n)) < beta, then the sample size is
            deemed to be sufficient to statistically distinguish these two
            populations and is marked as such. Else: The sample size is deemed
            to be insufficient.
    2. If the sample size (guess_) is sufficient to statistically distinguish
        the popuations, then the next sample size that is checked is
        guess_(n+1) = guess_(n) - guess_(n)/2. Else: The samples are deemed to
        be indistinguishable (i.e. too large of a sample is needed to
        distinguish them.) This prevents needlessly large sample sizes from
        being considered.
        A. Perform the same set of sample_draws comparisons and beta calculation
            as described above in (1)
    3. If the sample size (guess_) is sufficient to statistically distinguish
        the popuations, then the next sample size that is checked goes down by
        half of the difference between the last and current guess:
            guess_(n+1) = guess_(n) - [abs(guess_(n) - guess_(n-1))] / 2
        If guess_ was not sufficient to statistically distinguish the
        poplations, then the next sample size that is checked goes up by
        half of the difference between the last and current guess:
            guess_(n+1) = guess_(n) + [abs(guess_(n) - guess_(n-1))] / 2
        A. Perform the same set of sample_draws comparisons and beta calculation
            as described above in (1)
    4. Repeat 3 as needed until abs(guess_(n) - guess(n-1)) = 1. Then the true
    minimum sample size is whichever of {guess_(n), guess_(n-1)} is significant.

    Parameters
    ---------------
    fra_dist_1 : np.ndarray | .shape = (n)
        Vector of returned FRA values from population A (from generate_fra_
        distribution)
    likelihoods : np.ndarray | .shape = (obs, hyp)
        Likelihood function computed numerically. Used for LLR testing.
    ind_null_silent_truefrac : int 
        Index of null hypothesis in likelihoods (0 silent)
    init_guess : integer
        An initial guess for number of samples required to discriminate the
        two populations. The power analysis starts its search here. It is useful
        to start at intervals of 2^n. (Default 2048)
    sample_draws : integer
        How many paired samples to draw from both fra_dist_1 and fra_dist_2 to
        run paired statistical comparisons on during the Monte Carlo simulation.
    alpha : float |  0 < alpha < 1
        Alpha value for the power analysis (type 1 errors, false positives).
    beta : float |  0 < beta < 1
        Beta value for the power analysis (type 2 errors, false negatives)
    stat_test : string | 'ranksum' or 'chisquare'
        Specifies the type of statistical test to use to compare the populations
        of failure rate values.
    ctrl_n : False or integer
        If false, for paired comparisons both groups have the same sample size
        drawn from them. If integer, then one group has a fixed sample size
        (a control n, typically far less than the experimental group's n) while
        the other group has a sample size which varies as above throughout
        the algorithm until the
    verbosity : boolean
        Toggles verbose reporting of the estimated number of samples as the
        Monte Carlo simulation proceeds.

    Returns
    ------------
    min_samplesize_required : int
        The minimum number of samples required to reach the
        given alpha and beta levels of statistical confidence
    """

    # Initialize variables
    min_sample_reached = False
    current_guess = init_guess
    last_guess = 0
    last_correct = init_guess  # The last correct guess made.
    final_iter = False

    # Iterate through reductions in sample size
    while min_sample_reached is False:
        if verbosity is True:
            print('\t\tn_samples: ', str(current_guess), ' ... ', end='')

        if abs(current_guess - last_guess) < 1.5:
            final_iter = True

        # Pick indices for each sample set
        fra_subsample = np.random.choice(fra_dist_1,
                                         size=(sample_draws, current_guess))
        pvals_ = np.empty([sample_draws])

        # For each draw, compute the max-likelihood estimate
        for ind_ in range(fra_subsample.shape[0]):
            # Calculate likelihood-array-indices of samples by adding 200 to
            # align to -200:1:100 observations
            # Compute loglikelihood function given all observations
            ind_likelihood = (fra_subsample[ind_, :]*100+200).astype(np.int)
            loglikelihood_sum = np.sum(np.log(
                likelihoods[ind_likelihood, :]), axis=0)

            # Calculate argmax of likelihood
            ind_max_likelihood = np.argmax(loglikelihood_sum)
            if ind_ == 0 and plot_ex == True:
                plt.figure()
                plt.plot(loglikelihood_sum)
                plt.show()

            # Test statistic
            D = 2 * (loglikelihood_sum[ind_max_likelihood]
                     - loglikelihood_sum[ind_null_silent_truefrac])
            pval = sp_stats.distributions.chi2.sf(D, 1)
            pvals_[ind_] = pval

        # Calc fraction of draws where pvals_ reaches alpha significance
        fraction_alpha_signif = np.sum(pvals_ < alpha)

        # If beta significance not reached
        if (fraction_alpha_signif / len(pvals_)) > (1 - beta):
            if verbosity is True:
                print('*')

            last_guess_next = current_guess
            current_guess_next = round(
                current_guess - (abs(current_guess - last_guess) / 2))

            last_correct = current_guess

            last_guess = last_guess_next
            current_guess = current_guess_next

            if final_iter is True:
                min_sample_reached = True
                min_samplesize_required = last_correct
                print('\t\t\tn_final: ', str(last_correct))

                break

        # If beta significance reached
        elif (fraction_alpha_signif / len(pvals_)) < (1 - beta):
            if verbosity is True:
                print('NS')

            # Catch for beta significance being reached on the first guess
            if current_guess == init_guess:
                min_sample_reached = True
                min_samplesize_required = current_guess
                break

            last_guess_next = current_guess
            current_guess_next = round(
                current_guess + (abs(current_guess - last_guess) / 2))

            last_guess = last_guess_next
            current_guess = current_guess_next

            if final_iter is True:
                min_sample_reached = True
                min_samplesize_required = last_correct
                print('\t\t\tn_final: ', str(last_correct))

                break

    return min_samplesize_required


def _gen_fra_dist_fails(method='iterative',
                        pr_dist_sil=PrDist(sp_stats.uniform),
                        pr_dist_nonsil=PrDist(sp_stats.uniform),
                        silent_fraction=0.5,
                        num_trials=50, n_simulations=10000, n_start=100,
                        zeroing=False, graph_ex=False, verbose=False,
                        unitary_reduction=False, frac_reduction=0.2,
                        binary_vals=False, failrate_low=0.2,
                        failrate_high=0.8):
    """
    In-progress function to attempt to perform MLE using two variables per
    experiment: Fh and Fd. Here, the numerical simulations are performed and
    an estimate distribution along with failure rate distributions are returned.
    """

    if verbose is True:
        print('Generating FRA distribution with p(silent) = ',
              str(silent_fraction), ':')

    if binary_vals is True:
        fra_calc = np.zeros(n_simulations)
        fra_calc[0:int(silent_fraction * n_simulations)] = 1

        return fra_calc

    # First, generate realistic groups of neurons
    nonsilent_syn_group, silent_syn_group, \
        pr_nonsilent_syn_group, pr_silent_syn_group \
        = draw_subsample(method=method,
                         pr_dist_sil=pr_dist_sil,
                         pr_dist_nonsil=pr_dist_nonsil,
                         n_simulations=n_simulations,
                         n_start=n_start,
                         silent_fraction=silent_fraction,
                         failrate_low=failrate_low,
                         failrate_high=failrate_high,
                         unitary_reduction=unitary_reduction,
                         frac_reduction=frac_reduction,
                         verbose=verbose)

    # Calculate p(failure) mathematically for hyperpol,
    # depol based on product of (1- pr)
    math_failure_rate_hyperpol = np.ma.prod(
        1 - pr_nonsilent_syn_group, axis=1).compressed()

    pr_all_depol = np.ma.append(
        pr_nonsilent_syn_group, pr_silent_syn_group, axis=1)
    math_failure_rate_depol = np.ma.prod(
        1 - pr_all_depol, axis=1).compressed()

    # Simulate trials where failure rate is binary, calculate fraction fails
    sim_failure_rate_hyperpol = np.sum(
        np.random.rand(n_simulations, num_trials)
        < np.tile(math_failure_rate_hyperpol, (num_trials, 1)).transpose(),
        axis=1) / num_trials

    sim_failure_rate_depol = np.sum(
        np.random.rand(n_simulations, num_trials)
        < np.tile(math_failure_rate_depol, (num_trials, 1)).transpose(),
        axis=1) / num_trials

    # Calculate failure rate
    fra_calc = fra(sim_failure_rate_hyperpol,
                   sim_failure_rate_depol)

    # Filter out oddities
    fra_calc[fra_calc == -(np.inf)] = 0
    fra_calc[fra_calc == np.inf] = 0
    fra_calc[fra_calc == np.nan] = 0
    fra_calc = np.nan_to_num(fra_calc)

    if zeroing is True:
        fra_calc[fra_calc < 0] = 0

    return fra_calc, sim_failure_rate_hyperpol, sim_failure_rate_depol


def _mle_fh_fd():
    """
    Attempt to use pairs of observations (fh, fd) as inputs to a likelihood
    function for MLE.

    The idea would be that two obs. may provide better estimation ability than
    simply a single obs. (estimated failure rate through the FRA
    equation).

    However, as below shows, it is A) computationally intractable due to the
    fact that much more data has to be simulated to get smooth estimates for
    the frequency of (fh, fd) as a function of each hypothesis; and B) does not
    appear to provide significantly better estimation since fh is distributed
    identically across the hypothesis space.

    (However, for b), further work is probably needed...)

    """
    n_simulations = 20000
    obs_bins = 0.02
    zeroing = False

    hyp = np.linspace(0, 0.5, num=200)
    fra_calc = np.empty(len(hyp), dtype=np.ndarray)
    fra_fail_h = np.empty(len(hyp), dtype=np.ndarray)
    fra_fail_d = np.empty(len(hyp), dtype=np.ndarray)

    print('Generating FRA calcs...')
    for ind, silent in enumerate(hyp):
        print('\tSilent frac: ', str(silent))
        # Generate an FRA dist
        fra_calc[ind], fra_fail_h[ind], fra_fail_d[ind] \
            = _gen_fra_dist_fails(n_simulations=n_simulations,
                                  silent_fraction=silent, zeroing=zeroing)

    # 2. Create loglikelihood function for each case
    obs_fh = np.arange(0, 1, obs_bins)  # Bin observations from -200 to 100
    obs_fd = np.arange(0, 1, obs_bins)  # Bin observations from -200 to 100

    likelihood = np.empty([len(obs_fh), len(obs_fd), len(hyp)])

    for ind_obs_fh, obs_fh_ in enumerate(obs_fh):
        for ind_obs_fd, obs_fd_ in enumerate(obs_fd):
            for ind_hyp, hyp_ in enumerate(hyp):
                obs_in_range_fh_ = np.where(
                    np.abs(fra_fail_h[ind_hyp] - obs_fh_) < obs_bins/2)[0]
                obs_in_range_fd_ = np.where(
                    np.abs(fra_fail_d[ind_hyp] - obs_fd_) < obs_bins/2)[0]

                obs_in_range_both = set(obs_in_range_fh_) \
                    - (set(obs_in_range_fh_) - set(obs_in_range_fd_))

                p_obs = len(obs_in_range_both) / len(fra_fail_h[ind_hyp])

                likelihood[ind_obs_fh, ind_obs_fd, ind_hyp] = p_obs
    likelihood += 0.0001
    # (Set likelihoods microscopically away from 0 to avoid log(0) errors)

    # -----------------------------
    # Plotting
    # -----------------------------
    plt.style.use('presentation_ml')

    fig = plt.figure(figsize=(10, 6))
    spec = gridspec.GridSpec(nrows=2, ncols=3,
                             height_ratios=[1, 0.2],
                             width_ratios=[0.2, 1, 1])

    p_silent_1 = 0.1
    ind_psil_1 = np.abs(hyp - p_silent_1).argmin()
    p_silent_2 = 0.4
    ind_psil_2 = np.abs(hyp - p_silent_2).argmin()

    ax_failcomp = fig.add_subplot(spec[0, 1])
    ax_failcomp.plot(fra_fail_h[ind_psil_1], fra_fail_d[ind_psil_1], '.',
                     color=plt.cm.RdBu(0.2), alpha=0.1)
    ax_failcomp.plot(fra_fail_h[ind_psil_2], fra_fail_d[ind_psil_1], '.',
                     color=plt.cm.RdBu(0.8), alpha=0.1)

    ax_failcomp_fd = fig.add_subplot(spec[0, 0])
    ax_failcomp_fh = fig.add_subplot(spec[1, 1])

    ax_failcomp_fh.hist(fra_fail_h[ind_psil_1], bins=50,
                        density=True, facecolor=plt.cm.RdBu(0.2), alpha=0.5)
    ax_failcomp_fh.hist(fra_fail_h[ind_psil_2], bins=50,
                        density=True, facecolor=plt.cm.RdBu(0.8), alpha=0.5)

    ax_failcomp_fd.hist(fra_fail_d[ind_psil_1],
                        orientation='horizontal', bins=50,
                        density=True, facecolor=plt.cm.RdBu(0.2), alpha=0.5)
    ax_failcomp_fd.hist(fra_fail_d[ind_psil_2],
                        orientation='horizontal', bins=50,
                        density=True, color=plt.cm.RdBu(0.8), alpha=0.5)
    ax_failcomp_fh.set_xlabel('$F_{h}$')
    ax_failcomp_fh.set_ylabel('pdf')
    ax_failcomp_fd.set_ylabel('$F_{d}$')
    ax_failcomp_fd.set_xlabel('pdf')

    # ---------
    ax_likelihood = fig.add_subplot(spec[:, 2])

    n_sims_ex = 20
    silent_frac_ex = 0.2

    fra_calc_ex, fra_fail_h_ex, fra_fail_d_ex \
        = _gen_fra_dist_fails(n_simulations=n_sims_ex,
                              silent_fraction=silent_frac_ex, zeroing=False)

    joint_llhood = np.zeros(len(hyp))
    for ind in range(len(fra_fail_h_ex)):
        ind_fh = np.abs(fra_fail_h_ex[ind] - obs_fh).argmin()
        ind_fd = np.abs(fra_fail_d_ex[ind] - obs_fd).argmin()

        llhood_ = np.log(likelihood[ind_fh, ind_fd, :])
        joint_llhood += llhood_
    joint_lhood = np.exp(joint_llhood)

    ax_likelihood.plot(hyp, joint_lhood)
    ax_likelihood.set_xlabel('silent fraction')
    ax_likelihood.set_ylabel('likelihood')
    ax_likelihood.set_xlim([0, 0.5])
