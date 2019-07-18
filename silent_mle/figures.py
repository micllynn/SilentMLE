import numpy as np
import scipy as sp
import scipy.stats as stats
import scipy.signal as sp_signal

import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from .core import *

def plot_fig1(figname = 'Figure1.pdf', fontsize = 8):

    plt.style.use('publication_pnas_ml')

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(6.5)
    fig.set_figwidth(3.43)
    plt.rc('font', size = fontsize)          # controls default text sizes

    #Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows = 5, ncols = 4, figure=fig)


    #Define spec for entire fig
    # spec_all = gridspec.GridSpec(nrows = 2, ncols = 3, height_ratios = [1, 1],
    #                              width_ratios = [1, 1, 1])

    ##########################
    #Plot example simulation
    ########################
    sim_ex = fig.add_subplot(spec_all[2, 2:4])

    currents = np.where(np.random.rand((50)) < 0.5, 0.0, -10.0)
    currents = np.append(currents, np.where(np.random.rand((50)) < 0.5, 0.0, 10.0))
    currents += (np.random.rand(len(currents)) - 0.5) * 2

    suc_ind = np.where(np.abs(currents) > 5)[0]
    fail_ind = np.where(np.abs(currents) < 5)[0]

    sim_ex.plot(suc_ind, currents[suc_ind], '.', color = [0, 0, 0])
    sim_ex.plot(fail_ind, currents[fail_ind], '.', color = [0.7, 0.7, 0.7])
    sim_ex.set_xlabel('Sweeps')
    sim_ex.set_xlim([0, 100])
    sim_ex.set_xticks([0, 50, 100])
    sim_ex.set_ylabel('Sim. current (pA)')
    sim_ex.set_ylim([-15, 15])
    sim_ex.set_yticks([-10, 0, 10])

    failrate_hyp = np.sum(currents[0:50] > -5) / 50
    failrate_dep = np.sum(currents[50:] < 5) / 50

    txt_hyp = 'Hyp. fail rate = {}'.format(failrate_hyp)
    txt_dep = 'Dep. fail rate = {}'.format(failrate_dep)
    estimate = (1 - np.log(failrate_hyp) / np.log(failrate_dep)) * 100
    txt_calc = 'Estimated silent = {0:.1f} %'.format(estimate)

    #sim_ex.textc
    sim_ex.text(0.55, 0.3, txt_dep, fontsize = fontsize - 2,
                color = [0.8, 0.2, 0.1, 0.7], transform = sim_ex.transAxes,
                horizontalalignment='left')
    sim_ex.text(0.5, 0.95, txt_calc, fontsize = fontsize-2, fontweight = 'bold',
                color = [0.3, 0.1, 0.3, 0.7], transform = sim_ex.transAxes,
                horizontalalignment='center')

    ########################
    #Plot distribution estimates
    ########################
    estim_dist = fig.add_subplot(spec_all[3, 0:2])
    ylim_max = 0.2

    fails_hyp = np.sum(np.random.rand(50, 10000) < 0.5, axis = 0) / 50
    fails_dep = np.sum(np.random.rand(50, 10000) < 0.5, axis = 0) / 50
    calc = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100

    estim_dist.hist(calc, bins = 25, weights = np.ones_like(
            calc) / len(calc), facecolor = [0, 0, 0, 0.3])
    estim_dist.set_xlabel('Estimated silent (%)')
    estim_dist.set_xlim([-150, 100])
    estim_dist.set_xticks([-150, -100, -50, 0, 50, 100])
    estim_dist.set_ylabel('pdf')
    estim_dist.set_ylim([0, ylim_max])
    estim_dist.set_yticks(np.linspace(0, ylim_max, 5))

    ########################
    #Plot dist. estimates where sweep num is changing
    ########################
    sweep_change = fig.add_subplot(spec_all[3, 2:4])
    sw_ch_inset = inset_axes(sweep_change, width='100%', height='35%', loc = 3,
                             bbox_to_anchor = (.17, .7, .5, .95),
                             bbox_transform = sweep_change.transAxes,
                             borderpad = 0)
    sw_ch_inset.patch.set_alpha(0)

    trials = [50, 100, 200, 500]

    calc = np.empty(len(trials), dtype = np.ndarray)
    calc_sd = np.empty(len(trials))

    for ind, trial in enumerate(trials):
        fails_hyp = np.sum(np.random.rand(trial, 10000) < 0.5, axis = 0) / trial
        fails_dep = np.sum(np.random.rand(trial, 10000) < 0.5, axis = 0) / trial
        calc[ind] = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100
        calc_sd[ind] = np.std(calc[ind])

        ratio_ = ind / (len(trials) -1)
        red_ = 1 / (1 + np.exp( -5 * (ratio_ - 0.5)))
        color_ = [red_, 0.2, 1 - red_]

        sweep_change.hist(calc[ind], bins = 25, weights = np.ones_like(
            calc[ind]) / len(calc[ind]), facecolor = color_, alpha = 0.3)

        sw_ch_inset.plot(trial, calc_sd[ind], 'o', color = color_, alpha = 0.4)

    sw_ch_inset.plot(trials, calc_sd, color = [0, 0, 0, 0.3])

    ylim_max = 0.2

    sweep_change.set_xlabel('Estimated silent (%)')
    sweep_change.set_xlim([-150, 60])
    sweep_change.set_xticks([-150, -100, -50, 0, 50])
    sweep_change.set_ylabel('pdf')
    sweep_change.set_ylim([0, ylim_max])
    sweep_change.set_yticks(np.linspace(0, ylim_max, 5))

    sw_ch_inset.set_xlabel('Sweep num.', fontsize = fontsize - 2)
    sw_ch_inset.set_xlim([0, 600])
    sw_ch_inset.set_xticks([50, 100, 200, 500])
    sw_ch_inset.set_ylabel('Estimate std. dev.', fontsize = fontsize - 2)
    sw_ch_inset.set_ylim([0,40])
    sw_ch_inset.set_yticks(np.linspace(0, 40, 3))
    sw_ch_inset.tick_params(axis = 'both', which = 'major', labelsize = fontsize - 3)


    ########################
    #Plot dist. estimates where synapse num. is changing
    ########################
    num_change = fig.add_subplot(spec_all[4, 2:4])
    n_ch_inset = inset_axes(num_change, width='100%', height='35%', loc = 3,
                             bbox_to_anchor = (.17, .7, .5, .95),
                             bbox_transform = num_change.transAxes,
                             borderpad = 0)
    n_ch_inset.patch.set_alpha(0)

    n_syn = np.arange(1, 7)

    calc = np.empty(len(n_syn), dtype = np.ndarray)
    calc_sd = np.empty(len(n_syn))

    for ind, n in enumerate(n_syn):
        pr = 1 - (0.5 ** (1 / n))

        fails_hyp = np.sum(np.random.rand(50, 10000) < (1 - pr) ** n, axis = 0) / 50
        fails_dep = np.sum(np.random.rand(50, 10000) < (1 - pr) ** n, axis = 0) / 50
        calc[ind] = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100
        calc_sd[ind] = np.std(calc[ind])

        ratio_ = ind / (len(trials) -1)
        red_ = 1 / (1 + np.exp( -6 * (ratio_ - 0.5)))
        color_ = [red_ , 0.1, 1 - red_]

        num_change.hist(calc[ind], bins = 25, weights = np.ones_like(
            calc[ind]) / len(calc[ind]), facecolor = color_, alpha = 0.15)

        n_ch_inset.plot(n, calc_sd[ind], 'o', color = color_, alpha = 0.4)

    n_ch_inset.plot(n_syn, calc_sd, color = [0, 0, 0, 0.3])

    ylim_max = 0.2
    num_change.set_xlabel('Estimated silent (%)')
    num_change.set_xlim([-150, 60])
    num_change.set_xticks([-150, -100, -50, 0, 50])
    num_change.set_ylabel('pdfs')
    num_change.set_ylim([0, ylim_max])
    num_change.set_yticks(np.linspace(0, ylim_max, 5))

    n_ch_inset.set_xlabel('n synapses', fontsize = fontsize - 2)
    n_ch_inset.set_xlim([1, 7])
    n_ch_inset.set_xticks([2, 4, 6 ])
    n_ch_inset.set_ylabel('Estimate std. dev.', fontsize = fontsize - 2)
    n_ch_inset.set_ylim([20,40])
    n_ch_inset.set_yticks(np.linspace(20, 40, 3))
    n_ch_inset.tick_params(axis = 'both', which = 'major', labelsize = fontsize - 3)

    plt.savefig(figname)

    return

def plot_fig1_S1(figname = 'Figure1_S1.pdf', fontsize = 9):

    fig = plt.figure()

    fig.set_figheight(8)
    fig.set_figwidth(8)
    plt.style.use('publication_ml')
    plt.rc('font', size = fontsize)

    #Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows = 3, ncols = 3)

    ###########################
    #Example failrate plots
    ##############################
    spec_failex = gridspec.GridSpecFromSubplotSpec(3, 1, spec_all[0, 1], hspace = 0.8)
    failex = []

    pr = [0.05, 0.5, 0.95]
    #color_pr = [[0.46, 0.63, 0.85], [0.3, 0.3, 0.3], [0.90, 0.49, 0.32]]
    color_pr = [[0.2, 0.4, 0.8], [0.3, 0.3, 0.3], [0.90, 0.49, 0.32]]

    for ind, pr_ in enumerate(pr):
        failex.append(fig.add_subplot(spec_failex[ind, 0]))

        currents = np.where(np.random.rand((50)) < (1 - pr_), 0.0, -10.0)
        currents += (np.random.rand(len(currents)) - 0.5) * 2

        suc_ind = np.where(np.abs(currents) > 5)[0]
        fail_ind = np.where(np.abs(currents) < 5)[0]

        failex[ind].plot(suc_ind, currents[suc_ind], '.', color = color_pr[ind],
              alpha = 0.6, markeredgewidth=0)
        failex[ind].plot(fail_ind, currents[fail_ind], '.', color = [0.7, 0.7, 0.7],
            alpha = 0.6, markeredgewidth=0)
        failex[ind].set_ylim([-12, 2])
        failex[ind].set_yticks([-10, 0])
        failex[ind].set_xlim([0, 50])
        failex[ind].spines['bottom'].set_visible(True)
        failex[ind].xaxis.set_ticks_position('bottom')
        failex[ind].set_xticks([0, 25, 50])

        text_ = 'Pr = {}'.format(pr_)

        failex[ind].text(0.95, 0.5, text_, fontsize = fontsize ,
              fontweight = 'bold', color = color_pr[ind], alpha = 0.6,
              transform = failex[ind].transAxes, horizontalalignment = 'right',
              verticalalignment = 'center')

        if ind is 2:
            failex[ind].set_xlabel('Sweeps')

        if ind is 1:
            failex[ind].set_ylabel('Sim. current (pA)')

    ##########################
    #Failure rate dists
    ########################
    fail_dist = fig.add_subplot(spec_all[0, 2])
    fd_inset = inset_axes(fail_dist, width='100%', height='100%', loc = 3,
                             bbox_to_anchor = (0.5, 0.75, .4, .25),
                             bbox_transform = fail_dist.transAxes,
                             borderpad = 0)
    fd_inset.patch.set_alpha(0)

    #Calculate the failrate dists f
    fails = np.empty(len(pr), dtype = np.ndarray)

    for ind, pr_ in enumerate(pr):

        fails[ind] = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis = 0) / 50
        fail_dist.hist(fails[ind], bins = 25, weights = np.ones_like(fails[ind]) / len(fails[ind]),
                       facecolor = color_pr[ind], alpha = 0.9)

    fail_dist.legend(labels = pr, title = 'Pr', loc = (0.15, 0.35))
    fail_dist.set_xlabel('failure rate')
    fail_dist.set_xlim([0, 1])
    fail_dist.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    fail_dist.set_ylabel('probability density')
    fail_dist.set_ylim([0, 0.3])
    fail_dist.set_yticks(np.linspace(0, 0.3, 7))


    #Calculate the failrate sd
    inter_ = 0.01
    pr_fine = np.arange(inter_, 1 - inter_, inter_)

    fails_sd = np.empty(len(pr_fine))
    calc_sd = np.empty(len(pr_fine))

    for ind, pr_ in enumerate(pr_fine):
        fails_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis = 0) / 50
        fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis = 0) / 50

        calc_ = (1 - np.log(fails_) / np.log(fails_dp_)) * 100
        calc_ = calc_[~np.isnan(calc_)]
        calc_ = calc_[~np.isinf(calc_)]

        fails_sd[ind] = np.std(fails_)
        calc_sd[ind] = np.std(calc_)

    fd_inset.plot(pr_fine, fails_sd, linewidth = 1.5, color = [0, 0, 0, 0.5])
    fd_inset.set_xlabel('Pr')
    fd_inset.set_ylabel('std dev')
    fd_inset.set_ylim([0, 0.08])
    fd_inset.set_xlim([0, 1])

    ########################
    #Silent synapse examples
    ########################
    spec_calcex = gridspec.GridSpecFromSubplotSpec(3, 1, spec_all[1, 0], hspace = 0.8)
    calcex = []

    for ind, pr_ in enumerate(pr):
        calcex.append(fig.add_subplot(spec_calcex[ind, 0]))

        #Simulate failures over one experiment
        fails_hyp_bool = np.random.rand(50) > pr_
        fails_dep_bool = np.random.rand(50) > pr_

        fails_hyp_I = np.where(fails_hyp_bool == True, 0.0, -10.0)
        fails_hyp_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        fails_dep_I = np.where(fails_dep_bool == True, 0.0, 10.0)
        fails_dep_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        #Estimate silent synapse frac
        fails_hyp = np.sum(fails_hyp_bool, axis = 0) / 50
        fails_dep = np.sum(fails_dep_bool, axis = 0) / 50
        calc_ = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100

        #Plot the simulated currents
        suc_ind_hyp = np.where(np.abs(fails_hyp_I) > 5)[0]
        fail_ind_hyp = np.where(np.abs(fails_hyp_I) < 5)[0]
        suc_ind_dep = np.where(np.abs(fails_dep_I) > 5)[0]
        fail_ind_dep = np.where(np.abs(fails_dep_I) < 5)[0]

        calcex[ind].plot(suc_ind_hyp, fails_hyp_I[suc_ind_hyp], '.',
              color = color_pr[ind], alpha = 0.7, markeredgewidth=0)
        calcex[ind].plot(fail_ind_hyp, fails_hyp_I[fail_ind_hyp], '.',
              color = [0.7, 0.7, 0.7], alpha = 0.7, markeredgewidth=0)
        calcex[ind].plot(suc_ind_dep + 50, fails_dep_I[suc_ind_dep], '.',
              color = color_pr[ind], alpha = 0.7, markeredgewidth=0)
        calcex[ind].plot(fail_ind_dep + 50, fails_dep_I[fail_ind_dep], '.',
              color = [0.7, 0.7, 0.7], alpha = 0.7, markeredgewidth=0)

        calcex[ind].set_ylim([-12, 12])
        calcex[ind].set_yticks([-10, 0, 10])
        calcex[ind].set_xlim([0, 100])
        calcex[ind].spines['bottom'].set_visible(True)
        calcex[ind].xaxis.set_ticks_position('bottom')
        calcex[ind].set_xticks([0, 25, 50, 75, 100])

        text_pr = 'Pr = {}'.format(pr_)
        text_sscalc = 'Est. silent = {0:.1f} %'.format(calc_)

        calcex[ind].text(0.05, 0.8, text_pr, fontsize = fontsize ,
              fontweight = 'bold', color = color_pr[ind], alpha = 0.6,
              transform = calcex[ind].transAxes, horizontalalignment = 'left',
              verticalalignment = 'center')
        calcex[ind].text(0.95, 0.2, text_sscalc, fontsize = fontsize - 2,
              fontweight = 'bold', color = 'k',
              transform = calcex[ind].transAxes, horizontalalignment = 'right',
              verticalalignment = 'center')

        if ind is 2:
            calcex[ind].set_xlabel('Sweeps')

        if ind is 1:
            calcex[ind].set_ylabel('Sim. current (pA)')

    ########################
    #Silent synapse estimate plot
    ########################
    est_dist = fig.add_subplot(spec_all[1, 1])
    est_cumul = fig.add_subplot(spec_all[1, 2])

    ed_inset = inset_axes(est_dist, width='100%', height='100%', loc = 3,
                             bbox_to_anchor = (0.5, 0.75, .4, .25),
                             bbox_transform = est_dist.transAxes,
                             borderpad = 0)
    ed_inset.patch.set_alpha(0)

    #Calculate the silent synapse dists
    for ind, pr_ in enumerate(pr):
        fails_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis = 0) / 50
        fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis = 0) / 50
        calc_ = ((1 - np.log(fails_) / np.log(fails_dp_)) * 100)
        calc_ = calc_[~np.isnan(calc_)]
        calc_ = calc_[~np.isinf(calc_)]

        est_dist.hist(calc_, bins = 30, weights = np.ones_like(calc_) / len(calc_),
                       facecolor = color_pr[ind], alpha = 0.5)

        est_cumul.hist(calc_, bins = 100, density = True, histtype = 'step',
                       cumulative = True, color = color_pr[ind], alpha = 0.6,
                       linewidth = 4)

    est_dist.legend(labels = pr, title = 'Pr', loc = (0.15, 0.35))
    est_dist.set_xlabel('estimated silent (%)')
    est_dist.set_xlim([-300, 100])
    est_dist.set_xticks([-300, -200, -100, 0, 100])
    est_dist.set_ylabel('probability density')
    est_dist.set_ylim([0, 0.4])
    est_dist.set_yticks(np.linspace(0, 0.4, 5))


    #Calculate the failrate sd
    ed_inset.plot(pr_fine, calc_sd, linewidth = 1.5, color = [0, 0, 0, 0.5])
    ed_inset.set_xlabel('Pr')
    ed_inset.set_ylabel('std dev')
    ed_inset.set_ylim([0, 150])
    ed_inset.set_xlim([0, 1])

    est_cumul.legend(labels = pr, title = 'Pr')
    est_cumul.set_xlabel('estimated silent (%)')
    est_cumul.set_xlim([-500, 100])
    est_cumul.set_xticks([-500, -300, -100, 100])
    est_cumul.set_ylabel('cumulative probability density')
    est_cumul.set_ylim([0, 1])
    est_cumul.set_yticks(np.linspace(0, 1, 5))


    ########################
    #multi-synapse failure rate
    ########################
    est_dist_multi = fig.add_subplot(spec_all[2, 1])
    est_dist_multi_failx = fig.add_subplot(spec_all[2, 2])

    n_syn = np.arange(1, 7)
    calc_syn = np.empty_like(n_syn, dtype = np.ndarray)
    color_syn = np.empty_like(n_syn, dtype = np.ndarray)
    failrate_syn = np.empty_like(n_syn, dtype = np.ndarray  )

    for ind, n in enumerate(n_syn):
        calc_syn[ind] = np.empty_like(pr_fine)
        failrate_syn[ind] = np.empty_like(pr_fine)

        #Pick a color
        ratio_ = ind / (len(n_syn) - 1)
        blue_ = 1 / (1 + np.exp(-5 * (ratio_ - 0.5)))
        color_ = [0, 0.25 + blue_/3, 0.7 - (blue_ / 2)]
        color_syn[ind] = color_

        for ind_pr, pr_ in enumerate(pr_fine):
            failrate_syn[ind][ind_pr] = (1 - pr_) ** n

            fails_ = np.sum(np.random.rand(50, 10000) < (1 - pr_) ** n, axis = 0) / 50
            fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_) ** n, axis = 0) / 50
            calc_ = ((1 - np.log(fails_) / np.log(fails_dp_)) * 100)
            calc_ = calc_[~np.isnan(calc_)]
            calc_ = calc_[~np.isinf(calc_)]

            calc_syn[ind][ind_pr] = np.std(calc_)

        #Postprocessing on the data
        find_range = np.where(np.logical_or(failrate_syn[ind] < 0.05,
                                             failrate_syn[ind] > 0.95))
        calc_syn[ind][find_range] = np.nan

        est_dist_multi.plot(pr_fine, calc_syn[ind], color = color_syn[ind], alpha = 0.5,
                             linewidth = 2)
        est_dist_multi_failx.plot(failrate_syn[ind], calc_syn[ind],
                                  color = color_syn[ind], alpha = 0.5, linewidth = 2)

    est_dist_multi.legend(labels = n_syn, title = 'Num. synapses')
    est_dist_multi.set_xlabel('release probability')
    est_dist_multi.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    est_dist_multi.set_ylabel('std dev of estimated silent (%)')
    est_dist_multi.set_ylim([0, 150])
    est_dist_multi.set_yticks(np.arange(0, 151, 25))

    est_dist_multi_failx.legend(labels = n_syn, title = 'Num. synapses')
    est_dist_multi_failx.set_xlabel('failure rate')
    est_dist_multi_failx.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    est_dist_multi_failx.set_ylabel('std dev of estimated silent (%)')
    est_dist_multi_failx.set_ylim([0, 150])
    est_dist_multi_failx.set_yticks(np.arange(0, 151, 25))

    plt.savefig(figname)


def plot_fig1_S2(figname = 'Figure1_S2.pdf', fontsize = 9):

    fig = plt.figure()

    fig.set_figheight(4)
    fig.set_figwidth(8)
    plt.style.use('publication_ml')
    plt.rc('font', size = fontsize)

    #Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows = 3, ncols = 2, hspace=1)

    ###########################
    #Example plots
    ##############################
    failex = []

    fails_hyp = [0.45, 0.05, 0.85]
    fails_dep = [0.55, 0.15, 0.95]

    color_fails = np.empty(len(fails_hyp), dtype = np.ndarray)

    for ind, fails_hyp_ in enumerate(fails_hyp):

        failex.append(fig.add_subplot(spec_all[ind, 0]))

        #First, calculate the color to use
        ratio_ = ind / (len(fails_hyp) - 1)
        blue_ = 1 / (1 + np.exp(-5 * (ratio_ - 0.5)))
        color_ = [0, 0.25 + blue_/3, 0.7 - (blue_ / 2)]
        color_fails[ind] = color_

        #Simulate failures over one experiment
        fails_hyp_bool = np.zeros(50, dtype = np.bool)
        fails_hyp_bool[0:int(fails_hyp[ind] * 50)] = True
        np.random.shuffle(fails_hyp_bool)

        fails_dep_bool = np.zeros(50, dtype = np.bool)
        fails_dep_bool[0:int(fails_dep[ind] * 50)] = True
        np.random.shuffle(fails_dep_bool)

        fails_hyp_I = np.where(fails_hyp_bool == True, 0.0, -10.0)
        fails_hyp_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        fails_dep_I = np.where(fails_dep_bool == True, 0.0, 10.0)
        fails_dep_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        #Estimate silent synapse frac
        calc_ = (1 - np.log(fails_hyp[ind]) / np.log(fails_dep[ind])) * 100

        #Plot the simulated currents
        suc_ind_hyp = np.where(np.abs(fails_hyp_I) > 5)[0]
        fail_ind_hyp = np.where(np.abs(fails_hyp_I) < 5)[0]
        suc_ind_dep = np.where(np.abs(fails_dep_I) > 5)[0]
        fail_ind_dep = np.where(np.abs(fails_dep_I) < 5)[0]

        failex[ind].plot(suc_ind_hyp, fails_hyp_I[suc_ind_hyp], '.',
              color = color_fails[ind], alpha = 0.7, markeredgewidth=0)
        failex[ind].plot(fail_ind_hyp, fails_hyp_I[fail_ind_hyp], '.',
              color = [0.7, 0.7, 0.7], alpha = 0.7, markeredgewidth=0)
        failex[ind].plot(suc_ind_dep + 50, fails_dep_I[suc_ind_dep], '.',
              color = color_fails[ind], alpha = 0.7, markeredgewidth=0)
        failex[ind].plot(fail_ind_dep + 50, fails_dep_I[fail_ind_dep], '.',
              color = [0.7, 0.7, 0.7], alpha = 0.7, markeredgewidth=0)

        failex[ind].set_ylim([-16, 16])
        failex[ind].set_yticks([-10, 0, 10])
        failex[ind].set_xlim([-15, 100])
        failex[ind].spines['bottom'].set_visible(True)
        failex[ind].xaxis.set_ticks_position('bottom')
        failex[ind].set_xticks([0, 25, 50, 75, 100])

        if ind is 1:
            failex[ind].set_ylabel('Sim. current (pA)')
        if ind is 2:
            failex[ind].set_xlabel('Sweeps')

        text_hp = 'hyp. fails: {0:.2f}'.format(fails_hyp[ind])
        text_dp = 'dep. fails: {0:.2f}'.format(fails_dep[ind])

        failex[ind].text(0.05, 0.8, text_hp, fontsize = fontsize,
              fontweight = 'bold', color = color_fails[ind], alpha = 0.4,
              transform = failex[ind].transAxes, horizontalalignment = 'left',
              verticalalignment = 'center')
        failex[ind].text(0.95, 0.2, text_dp, fontsize = fontsize,
              fontweight = 'bold', color = color_fails[ind], alpha = 0.9,
              transform = failex[ind].transAxes, horizontalalignment = 'right',
              verticalalignment = 'center')

    ###################
    #Natural logarithm
    spl_log = fig.add_subplot(spec_all[:, 1])

    fails_fine = np.linspace(0, 1, num = 1000)
    log_fails_fine = np.log(fails_fine)
    spl_log.plot(fails_fine, log_fails_fine, color = 'k',
                 linewidth = 2)
    spl_log.set_xlabel('fail rate')
    spl_log.set_ylabel('log(fail rate)')
    spl_log.set_ylim([-4, 1])

    for ind, fails_hyp_ in enumerate(fails_hyp):
        spl_log.annotate("",
                         xy = (fails_hyp[ind], np.log(fails_hyp[ind])),
                         xycoords = 'data',
                         xytext = (fails_hyp[ind], -4),
                         textcoords = 'data',
                         arrowprops = dict(arrowstyle="->",
                         connectionstyle="arc3", color = color_fails[ind],
                         alpha = 0.4, linewidth = 1.5, shrinkA=1, shrinkB=0.5))

        spl_log.annotate("",
                         xy = (fails_dep[ind], np.log(fails_dep[ind])),
                         xycoords = 'data',
                         xytext = (fails_dep[ind], -4),
                         textcoords = 'data',
                         arrowprops = dict(arrowstyle="->",
                         connectionstyle="arc3", color = color_fails[ind],
                         alpha = 0.9, linewidth = 1.5, shrinkA=1, shrinkB=0.5))

        spl_log.annotate("",
                         xy = (fails_dep[ind], -3.5),
                         xycoords = 'data',
                         xytext = (fails_hyp[ind], -3.5),
                         textcoords = 'data',
                         arrowprops = dict(arrowstyle="<->",
                         connectionstyle="arc3", color = 'k',
                         alpha = 0.6, linewidth = 1, shrinkA=0.5, shrinkB=0.5))

        xcoord_text_ = (fails_hyp[ind] + fails_dep[ind]) / 2
        ycoord_text_ = -3.75

        spl_log.text(xcoord_text_, ycoord_text_, '$\Delta$ 0.1', fontsize = fontsize - 3,
                     color = 'k', alpha = 0.6, horizontalalignment = 'center',
                     verticalalignment = 'center')


        #Fill in between the arrows
        xrange_fill = np.linspace(fails_hyp[ind], fails_dep[ind], num = 100)
        y1_fill = np.log(xrange_fill)
        y2_fill = np.ones_like(xrange_fill) * (np.log(fails_hyp[ind]) - 0.04)

        spl_log.fill_between(xrange_fill, y1_fill, y2_fill, facecolor = color_fails[ind],
                             alpha = 0.3)

        est_silent = (1 - np.log(fails_hyp[ind]) / np.log(fails_dep[ind])) * 100
        text_calc = '{0:.1f} % \n est. silent'.format(est_silent)

        #Make text reporting failure rates
        xcoord_text_ = (fails_hyp[ind] + fails_dep[ind]) / 2
        ycoord_text_ = np.log(fails_dep[ind]) + 0.6

        spl_log.text(xcoord_text_, ycoord_text_, text_calc, fontsize = fontsize,
              fontweight = 'bold', color = color_fails[ind], alpha = 1,
              horizontalalignment = 'center',
              verticalalignment = 'center')

    plt.show()
    fig.savefig(figname)

plot_fig2()


def plot_fig2(silent_fraction_low = 0.1, silent_fraction_high = 0.9,
              plot_sims = 40, fontsize = 9, ylab_pad_tight = -3,
              figname = 'Fig2.pdf',
              trueval_lim = 0.1, frac_reduction = 0.1, method_ = 'iterative',
              pr_dist = 'uniform'):

    '''
    Fig 2:
        a. Schematic of simulations
        b-c. Example simulations and sim. currents
        d-e. Description of n and p for silent and nonsilent synapses contained
            in each simulated set
        f. Example FRA histogram

        g. Estimator bias
        h. Estimator variance
    '''

    # silent_fraction_low = 0.1; silent_fraction_high = 0.9
    # plot_sims = 40; fontsize = 6; ylab_pad_tight = -3
    # figname = 'Fig2.pdf'
    # trueval_lim = 0.1; frac_reduction = 0.1; method_ = 'iterative'
    # pr_dist = 'uniform'

    #-------------------
    #1: Simulation of synapse groups
    #-------------------
    n_start = 200

    #Run simulation: 0.5 silent
    nonsilent_syn_group_half, silent_syn_group_half, \
    pr_nonsilent_syn_group_half, pr_silent_syn_group_half = \
    draw_subsample(method = method_, pr_dist = pr_dist,
                                     n_simulations = 10000, n_start = n_start,
                                     silent_fraction = 0.5, failrate_low = 0.2,
                                     failrate_high = 0.8,
                                     unitary_reduction = False,
                                     frac_reduction = frac_reduction)

    #Run simulation for the whole range
    silent_truefrac_coarse = np.arange(0.1, 0.95, 0.2)

    ratio_estimate_silent_array = np.empty(len(silent_truefrac_coarse), dtype = np.ndarray)

    ratio_estimate_silent_mean = np.empty(len(silent_truefrac_coarse))
    ratio_estimate_silent_std = np.empty(len(silent_truefrac_coarse))

    nonsilent_syn_group = np.empty(len(silent_truefrac_coarse), dtype = np.ndarray)
    silent_syn_group = np.empty_like(nonsilent_syn_group)
    pr_nonsilent_syn_group = np.empty_like(nonsilent_syn_group)
    pr_silent_syn_group = np.empty_like(nonsilent_syn_group)

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        print('\nSilent fraction: ', str(silent))

        nonsilent_syn_group[ind_silent], silent_syn_group[ind_silent], \
        pr_nonsilent_syn_group[ind_silent], pr_silent_syn_group[ind_silent] \
        = draw_subsample(method = method_, pr_dist = pr_dist,
                                           n_simulations = 10000, n_start = n_start,
                                           silent_fraction = silent, failrate_low = 0.2,
                                           failrate_high = 0.8,
                                           unitary_reduction = False,
                                           frac_reduction = frac_reduction)

        ratio_estimate_silent_array[ind_silent] = silent_syn_group[ind_silent] / (
                nonsilent_syn_group[ind_silent] + silent_syn_group[ind_silent])

        ratio_estimate_silent_mean[ind_silent] = np.mean(ratio_estimate_silent_array[ind_silent])
        ratio_estimate_silent_std[ind_silent] = np.std(ratio_estimate_silent_array[ind_silent])


    #-------------------
    #2: Simulation of FRA values returned
    #-------------------
    silent_truefrac_fine = np.arange(0, 0.55, 0.01)

    fra_calc = np.empty(len(silent_truefrac_fine), dtype = np.ndarray)
    fra_calc_z = np.empty(len(silent_truefrac_fine), dtype = np.ndarray) #zeroing

    #Store fraction of sims reaching true val, and the error of the mean sims
    #in small samples of n = 10, 20, 30.
    frac_true = np.empty(len(silent_truefrac_fine))
    mean_fra_n10 = np.empty(len(silent_truefrac_fine))
    mean_fra_n20 = np.empty(len(silent_truefrac_fine))
    mean_fra_n30 = np.empty(len(silent_truefrac_fine))
    std_fra_n10 = np.empty(len(silent_truefrac_fine))
    std_fra_n20 = np.empty(len(silent_truefrac_fine))
    std_fra_n30 = np.empty(len(silent_truefrac_fine))

    frac_true_z = np.empty(len(silent_truefrac_fine))
    mean_fra_n10_z = np.empty(len(silent_truefrac_fine))
    mean_fra_n20_z = np.empty(len(silent_truefrac_fine))
    mean_fra_n30_z = np.empty(len(silent_truefrac_fine))
    std_fra_n10_z = np.empty(len(silent_truefrac_fine))
    std_fra_n20_z = np.empty(len(silent_truefrac_fine))
    std_fra_n30_z = np.empty(len(silent_truefrac_fine))

    for ind, silent in enumerate(silent_truefrac_fine):
        print('\nSilent fraction: ', str(silent))

        #Make calculation of FRA values
        fra_calc[ind] = gen_fra_dist(method = method_,
                                             pr_dist = pr_dist,
                                             n_simulations = 10000,
                                             silent_fraction = silent,
                                             zeroing = False,
                                             unitary_reduction = False,
                                             frac_reduction = frac_reduction)

        fra_calc_z[ind] = gen_fra_dist(method = method_,
                                             pr_dist = pr_dist,
                                             n_simulations = 10000,
                                             silent_fraction = silent,
                                             zeroing = True,
                                             unitary_reduction = False,
                                             frac_reduction = frac_reduction)

        #Store fraction of 'true' estimates
        error_ = np.abs(fra_calc[ind] - silent_truefrac_fine[ind])
        frac_true[ind] = len(np.where(error_ < trueval_lim)[0]) / len(error_)

        error_z_ = np.abs(fra_calc_z[ind] - silent_truefrac_fine[ind])
        frac_true_z[ind] = len(np.where(error_z_ < trueval_lim)[0]) / len(error_)


        #Draw subsets of 10, 20, 30 samples from FRA and store means
        draws_n10_ = np.empty(1000)
        draws_n20_ = np.empty(1000)
        draws_n30_ = np.empty(1000)

        draws_n10_z_ = np.empty(1000)
        draws_n20_z_ = np.empty(1000)
        draws_n30_z_ = np.empty(1000)

        for draw in range(1000):
            inds_n10_ = np.random.randint(0, 10000, size = 10)
            inds_n20_ = np.random.randint(0, 10000, size = 20)
            inds_n30_ = np.random.randint(0, 10000, size = 30)

            draws_n10_[draw] = np.mean(fra_calc[ind][inds_n10_])
            draws_n20_[draw] = np.mean(fra_calc[ind][inds_n20_])
            draws_n30_[draw] = np.mean(fra_calc[ind][inds_n30_])

            draws_n10_z_[draw] = np.mean(fra_calc_z[ind][inds_n10_])
            draws_n20_z_[draw] = np.mean(fra_calc_z[ind][inds_n20_])
            draws_n30_z_[draw] = np.mean(fra_calc_z[ind][inds_n30_])

        mean_fra_n10[ind] = np.mean(draws_n10_ - silent)
        mean_fra_n20[ind] = np.mean(draws_n20_ - silent)
        mean_fra_n30[ind] = np.mean(draws_n30_ - silent)

        mean_fra_n10_z[ind] = np.mean(draws_n10_z_ - silent)
        mean_fra_n20_z[ind] = np.mean(draws_n20_z_ - silent)
        mean_fra_n30_z[ind] = np.mean(draws_n30_z_ - silent)

        std_fra_n10[ind] = np.std(draws_n10_ - silent)
        std_fra_n20[ind] = np.std(draws_n20_ - silent)
        std_fra_n30[ind] = np.std(draws_n30_ - silent)

        std_fra_n10_z[ind] = np.std(draws_n10_z_ - silent)
        std_fra_n20_z[ind] = np.std(draws_n20_z_ - silent)
        std_fra_n30_z[ind] = np.std(draws_n30_z_ - silent)


    #-----------------------------PLOTTING------------------------------------
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(6.5)
    fig.set_figwidth(3.43)
    plt.style.use('publication_pnas_ml')
    fontsize = 8
    plt.rc('font', size = fontsize)          # controls default text sizes

    #Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows = 6, ncols = 6, figure=fig)

    #Define colors to be used
    color_nonsilent = np.array([0.1, 0.55, 0.85])
    color_silent = np.array([0.76, 0.26, 0.22])

    img_colormod_ns = 0.4
    img_colormod_s = 1.1
    img_alpha = 0.5

    color_pr = np.array([0.25, 0.55, 0.18])

    color_ratio_accuracy = [ 0.37,  0.25,  0.21]


    #Subplot 2: Number of active/silent synapses
    n_plot_nonsil = fig.add_subplot(spec_all[0, 2:4])
    n_plot_sil = fig.add_subplot(spec_all[0, 4:6], sharey = n_plot_nonsil)
    #ratio_accuracy = fig.add_subplot(spec_bottomleft[:, 2])

    count_nonsil = np.empty(len(silent_truefrac_coarse), dtype = np.ndarray)
    count_sil = np.empty(len(silent_truefrac_coarse), dtype = np.ndarray)

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        count_nonsil[ind_silent] = np.bincount(nonsilent_syn_group[ind_silent])
        count_nonsil[ind_silent] = count_nonsil[ind_silent] / np.sum(count_nonsil[ind_silent])

        count_sil[ind_silent] = np.bincount(silent_syn_group[ind_silent])
        count_sil[ind_silent] = count_sil[ind_silent] / np.sum(count_sil[ind_silent])

        n_plot_nonsil.plot(np.arange(1, len(count_nonsil[ind_silent])), count_nonsil[ind_silent][1:],
                           color = color_nonsilent * silent, alpha = 0.9, linewidth = 1.5)
        n_plot_sil.plot(np.arange(0, len(count_sil[ind_silent])), count_sil[ind_silent],
                        color = color_silent * silent, alpha = 0.9, linewidth = 1.5)


    n_plot_nonsil.set_xlabel('active synapses')
    n_plot_nonsil.set_ylabel('probability density')
    n_plot_nonsil.set_xticks(np.arange(0, 9, 2))
    n_plot_nonsil.set_xlim([0, 8])

    n_plot_nonsil.set_ylim([0, 0.8])
    #n_plot_nonsil.set_title('n: Active', alpha = 0.5, loc = 'left')
    leg_ns = n_plot_nonsil.legend(labels = (silent_truefrac_coarse * 100).astype(np.int),
                         title = 'silent (%)', ncol = 2, loc = 1,
                         handlelength = 0.5, fontsize = 'x-small')

    n_plot_sil.set_xlabel('silent synapses')
    #n_plot_sil.set_ylabel('probability density')
    n_plot_sil.set_xticks(np.arange(0, 50, 10))
    n_plot_sil.set_xlim([0, 50])
    n_plot_sil.set_ylim([0, 0.8])
    #n_plot_sil.set_title('n: Silent', alpha = 0.5, loc = 'left')
    leg_s = n_plot_sil.legend(labels = (silent_truefrac_coarse * 100).astype(np.int),
                      title = 'silent (%)', ncol = 2, loc = 1,
                      handlelength = 0.5, fontsize = 'x-small')

    #Subplot 4/5: Nonsil/sil Release probability dist.:
    release_prob_nonsil = fig.add_subplot(spec_all[1, 2:4])
    subset_error = fig.add_subplot(spec_all[1, 4:6])

    max_n_nonsilent = len(np.bincount(nonsilent_syn_group_half))
    max_n_silent = len(np.bincount(silent_syn_group_half))

    pr_nonsil = np.empty(max_n_nonsilent, dtype = np.ndarray)
    n_pr_toplot = 500

    #Plot mean release probability for each number of nonsilent/silent syns
    for syn_number in range(1, max_n_nonsilent):
        inds_with_syn_number = np.where(pr_nonsilent_syn_group_half
                                        .count(axis = 1) == syn_number)
        pr_nonsil[syn_number] = np.ma.mean(
                pr_nonsilent_syn_group_half[inds_with_syn_number], axis = 1)

        if pr_nonsil[syn_number] is not None:
            release_prob_nonsil.plot(
                    np.ones(len(pr_nonsil[syn_number][0:n_pr_toplot]))* syn_number,
                    pr_nonsil[syn_number][0:n_pr_toplot], '.', color = color_pr,
                    alpha = 0.1, markersize = 1)

    #Plot the subset composition vs superset composition for each position
    subset_fracsil = np.empty(len(silent_truefrac_coarse), dtype = np.ndarray)
    subset_fracsil_mean =  np.empty(len(silent_truefrac_coarse))
    subset_fracsil_sd =  np.empty(len(silent_truefrac_coarse))

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        subset_fracsil[ind_silent] = silent_syn_group[ind_silent] \
            / (nonsilent_syn_group[ind_silent] + silent_syn_group[ind_silent])

        subset_fracsil_mean[ind_silent] = np.mean(subset_fracsil[ind_silent])
        subset_fracsil_sd[ind_silent] = np.std(subset_fracsil[ind_silent])

    subset_error.plot(silent_truefrac_coarse, subset_fracsil_mean,
       color = color_silent, alpha = 0.9, linewidth = 1.5)
    subset_error.fill_between(silent_truefrac_coarse,
        subset_fracsil_mean - subset_fracsil_sd,
        subset_fracsil_mean + subset_fracsil_sd,
        facecolor = color_silent, alpha = 0.2)

    subset_error.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10), '--k',
        linewidth = 0.5)

    #Adjust plot features
    release_prob_nonsil.set_xlim([0, 8.3])
    release_prob_nonsil.set_ylim([0, 1])
    release_prob_nonsil.set_xticks(np.arange(0, 9, 2))
    release_prob_nonsil.set_yticks([0, 0.5, 1])
    release_prob_nonsil.set_xlabel('active synapses')
    release_prob_nonsil.set_ylabel('mean Pr')

    subset_error.set_xlim([0, 1])
    subset_error.set_ylim([0, 1])
    subset_error.set_xticks([0, 0.5, 1])
    subset_error.set_yticks([0, 0.5, 1])
    subset_error.set_xlabel('frac. silent')
    subset_error.set_ylabel('subset frac. sil.')

    # -------------------
    # 1: Plot example synapse groups and example traces
    # -------------------
    sim_ex_30 = fig.add_subplot(spec_all[2:3, 0:3])
    #Subplot 1: synapsegroups
    img_groups_30 = np.ones((100, np.max(nonsilent_syn_group[1][0:plot_sims]
    + silent_syn_group[1][0:plot_sims]), 3))

    for sim in range(plot_sims):
        img_groups_30[sim, 0:nonsilent_syn_group[1][sim], :] = np.tile(
                [0, 0, 0], (1, nonsilent_syn_group[1][sim], 1))
        img_groups_30[sim, nonsilent_syn_group[1][sim] :
            nonsilent_syn_group[1][sim] + silent_syn_group[1][sim], :] = np.tile(
                [0.8, 0.1, 0.2], (1, silent_syn_group[1][sim], 1))
    #Plot 30% silent example
    img30 = sim_ex_30.imshow(img_groups_30, aspect = 'auto', alpha = 0.8,
        interpolation = 'nearest')
    sim_ex_30.plot()
    sim_ex_30.set_ylabel('simulations')
    sim_ex_30.set_ylim([plot_sims - 0.5, -0.5])
    sim_ex_30.set_yticks([])
    sim_ex_30.set_xticks([0 - 0.5, 5 - 0.5, 10 - 0.5, 15 - 0.5])
    sim_ex_30.set_xticklabels([])
    sim_ex_30.text(1, 0.55, 'nonsilent', verticalalignment = 'top',
                       horizontalalignment = 'right', color = [0, 0, 0],
                       transform = sim_ex_30.transAxes,
                       fontsize = fontsize - 1, fontweight = 'bold')
    sim_ex_30.text(1, 0.45, 'silent', verticalalignment = 'top',
                       horizontalalignment = 'right', color = [0.8, 0.1, 0.2],
                       transform = sim_ex_30.transAxes,
                       fontsize = fontsize - 1, fontweight = 'bold')

    # -------------------
    # 2: Example traces for various silent fracs
    # #-------------------
    silent_extraces = [0.3]
    failex = []

    for ind, silent_ in enumerate(silent_extraces):
        failex.append(fig.add_subplot(spec_all[3:4, 0:3]))

        nonsilent_syn_group_, silent_syn_group_, \
        pr_nonsilent_syn_group_, pr_silent_syn_group_ \
        = draw_subsample(pr_dist = pr_dist,
                                           n_simulations = 1, n_start = n_start,
                                           silent_fraction = silent_, failrate_low = 0.2,
                                           failrate_high = 0.8,
                                           unitary_reduction = False,
                                           frac_reduction = frac_reduction)

        pr_sweeps_hyp_ = np.tile(pr_nonsilent_syn_group_.compressed(), 50).reshape(50,
                            nonsilent_syn_group_[0])
        pr_sweeps_dep_ = np.tile(np.append(pr_nonsilent_syn_group_.compressed(),
                            pr_silent_syn_group_.compressed()), 50).reshape(50,
                            nonsilent_syn_group_[0] + silent_syn_group_[0])

        currents_hyp = np.where(np.random.rand(50, nonsilent_syn_group_[0]) <
                                           (1 - pr_sweeps_hyp_), 0.0, -10.0)
        currents_hyp += (np.random.rand(currents_hyp.shape[0], currents_hyp.shape[1])
                        - 0.5) * 2
        currents_hyp_sum = np.sum(currents_hyp, axis = 1)

        currents_dep = np.where(np.random.rand(50, nonsilent_syn_group_[0] +
                                               silent_syn_group_[0]) <
                                               (1 - pr_sweeps_dep_), 0.0, 10.0)
        currents_dep += (np.random.rand(currents_dep.shape[0], currents_dep.shape[1])
                        - 0.5) * 2
        currents_dep_sum = np.sum(currents_dep, axis = 1)


        suc_ind_hyp = np.where(np.abs(currents_hyp_sum) > 5)[0]
        fail_ind_hyp = np.where(np.abs(currents_hyp_sum) < 5)[0]

        suc_ind_dep = np.where(np.abs(currents_dep_sum) > 5)[0]
        fail_ind_dep = np.where(np.abs(currents_dep_sum) < 5)[0]

        est_silent = (1 - np.log(len(fail_ind_hyp) / 50) / np.log(len(fail_ind_dep) / 50)) * 100


        failex[ind].plot(suc_ind_hyp, currents_hyp_sum[suc_ind_hyp], '.',
              color = color_silent * 0.5,
              alpha = 0.6, markeredgewidth=0)
        failex[ind].plot(fail_ind_hyp, currents_hyp_sum[fail_ind_hyp], '.', color = [0.7, 0.7, 0.7],
            alpha = 0.8, markeredgewidth=0)
        failex[ind].plot(suc_ind_dep + 50, currents_dep_sum[suc_ind_dep], '.',
              color = color_nonsilent * 0.5,
              alpha = 0.6, markeredgewidth=0)
        failex[ind].plot(fail_ind_dep + 50, currents_dep_sum[fail_ind_dep], '.', color = [0.7, 0.7, 0.7],
            alpha = 0.8, markeredgewidth=0)
        failex[ind].plot([0, 100], [0, 0], ':k', linewidth = 1)


#        failex[ind].set_ylim([-12, 2])
#        failex[ind].set_yticks([-10, 0])
        failex[ind].set_xlim([0, 100])
        failex[ind].set_xticks([0, 25, 50, 75, 100])
        ymax_ = failex[ind].get_ylim()[1]
        failex[ind].set_ylim([-ymax_, ymax_])
        failex[ind].spines['bottom'].set_visible(True)
        failex[ind].xaxis.set_ticks_position('bottom')
        failex[ind].set_ylabel('sim. current (pA)')
        failex[ind].set_xlabel('sweeps')

        text_truesil = '{}% silent'.format(int(silent_ * 100))
        text_estsil = 'Est. silent\n = {0:.1f}%'.format(est_silent)

        failex[ind].text(0.05, 0.95, text_truesil, fontsize = fontsize + 1,
              fontweight = 'bold', color = color_nonsilent * 0.4, alpha = 0.8,
              transform = failex[ind].transAxes, horizontalalignment = 'left',
              verticalalignment = 'top',
              bbox=dict(facecolor='none',
                        edgecolor= np.append(color_nonsilent * 0.5, 0.8), pad=2.0))

        failex[ind].text(1, 0.1, text_estsil, fontsize = fontsize - 1 ,
              fontweight = 'bold', color = [0, 0, 0], alpha = 1,
              transform = failex[ind].transAxes, horizontalalignment = 'right',
              verticalalignment = 'bottom')




    #-------------------
    #3: Plots of FRA values returned
    #-------------------

    #########
    #first, plot estimate distributions
    ######
    histtype_ = 'step'
    desired_silent = 0.3
    ind_desired_silent = np.argmin(np.abs(silent_truefrac_fine - desired_silent))

    ax_frahist_z = fig.add_subplot(spec_all[2:4, 3:6])

    #Histogram
    ax_frahist_z.hist(fra_calc_z[0] * 100, bins = 20, weights = np.ones_like(
            fra_calc_z[0]) / len(fra_calc_z[0]), color = [0, 0, 0, 1],
            histtype = histtype_, linewidth = 2)
    ax_frahist_z.hist(fra_calc_z[ind_desired_silent] * 100, bins = 20, weights = np.ones_like(
            fra_calc_z[ind_desired_silent]) / len(fra_calc_z[ind_desired_silent]), color = color_silent * 0.8,
            alpha = 0.8, histtype = histtype_, linewidth = 2)
    ax_frahist_z.legend(labels = ['0', str(desired_silent * 100)], title = 'silent (%)')
    #ax_frahist_z.set_title('Returned estimates (zeroing)', alpha = 0.5, loc = 'left')
    ax_frahist_z.set_xlim([0, 100])
    ax_frahist_z.set_xlabel('estimated silent (%)')
    ax_frahist_z.set_ylabel('probability density')

    #-------------------
    #4: Bias and variance of estimator
    #-------------------
    ax_bias = fig.add_subplot(spec_all[4:6, 0:3])

    bias = np.empty(len(silent_truefrac_fine))
    bias_no_z = np.empty(len(silent_truefrac_fine))
    for ind, frac in enumerate(silent_truefrac_fine):
        bias[ind] = (np.mean(fra_calc_z[ind]) - silent_truefrac_fine[ind])
        bias_no_z[ind] = (np.mean(fra_calc[ind]) - silent_truefrac_fine[ind])
    ax_bias.plot(silent_truefrac_fine * 100, bias * 100, color = [0, 0, 0])
    ax_bias.plot(silent_truefrac_fine * 100, bias_no_z * 100, color = [0.5, 0.5, 0.5])
    ax_bias.legend(labels = ['FRA', 'FRA (neg.)'], title = 'estimator')
    ax_bias.set_xlabel('ground truth silent (%)')
    ax_bias.set_ylabel('estimator bias (%)')
    ax_bias.set_ylim([0, ax_bias.get_ylim()[1]])
    ax_bias.set_xticks([0, 10, 20, 30, 40, 50])


    ax_var = fig.add_subplot(spec_all[4:6, 3:6])

    var = np.empty(len(silent_truefrac_fine))
    var_no_z = np.empty(len(silent_truefrac_fine))
    for ind, frac in enumerate(silent_truefrac_fine):
        var[ind] = np.var(fra_calc_z[ind]*100)
        var_no_z[ind] = np.var(fra_calc[ind]*100)
    ax_var.plot(silent_truefrac_fine * 100, var, color = [0, 0, 0])
    ax_var.plot(silent_truefrac_fine * 100, var_no_z, color = [0.5, 0.5, 0.5])
    ax_var.legend(labels = ['FRA', 'FRA (neg.)'], title = 'estimator')
    ax_var.set_xlabel('ground truth silent (%)')
    ax_var.set_ylabel('estimator var. (%)')
    ax_var.set_ylim([0, ax_var.get_ylim()[1]])
    ax_var.set_xticks([0, 10, 20, 30, 40, 50])

    #Set tight layouts for all
    spec_all.tight_layout(fig)
    plt.savefig(figname)



def plot_fig4(n_true_silents = 26, fontsize = 12, sample_draws = 5000,
              figname = 'Fig4.pdf'):
    '''
    Plot the power analysis figure

    '''

    ########################################################################
    #1. Simulations
    ########################################################################
    #Uncomment code for within-function segment running and troubleshootings
    n_true_silents = 20
    fontsize = 12
    sample_draws = 1000
    figname = 'Fig4.pdf'

    silent_truefrac = np.linspace(0, 0.5, num = n_true_silents)
    fra_calc = np.empty(len(silent_truefrac), dtype = np.ndarray)
    binary_calc = np.empty_like(fra_calc)

    #Set up multiple conditions
    conditions = ['base', 'c10', 'c20', 'binary', 'llr', 'llr_binary']

    #Set parameters for each condition
    conds_beta = {cond: 0.2 for cond in conditions}

    conds_ctrl_n = {cond: False for cond in conditions}
    conds_ctrl_n['c10'] = 10
    conds_ctrl_n['c20'] = 20

    mins_templ = np.empty((len(silent_truefrac), len(silent_truefrac)))
    mins_1d_templ = np.empty((len(silent_truefrac)))

    minsamples = {cond : mins_templ.copy() for cond in conditions[0:4]}
    for cond in conditions[4:]:
        minsamples[cond] = mins_1d_templ.copy() #Set minsamples for llr to be 1d


    print('Generating FRA calcs...')
    #Generate FRA calcs and calc simple minsamples versus baseline
    for ind, silent in enumerate(silent_truefrac):
        #Generate an FRA dist
        fra_calc[ind] = gen_fra_dist(n_simulations = 10000,
                                             silent_fraction = silent,
                                             zeroing = True,
                                             unitary_reduction = False,
                                             frac_reduction = 0.2)

        binary_calc[ind] = gen_fra_dist(n_simulations = 10000,
                                             silent_fraction = silent,
                                             zeroing = True,
                                             unitary_reduction = False,
                                             frac_reduction = 0.2,
                                             binary_vals = True)

    #Set up guesses for each condition to start at
    guess = {cond : int(2048) for cond in conditions}

    #------------------
    #Minsamples 1: Standard FRA with numerical sims
    for ind1_sil, sil1 in enumerate(silent_truefrac):
        for ind2_sil , sil2 in enumerate(silent_truefrac):

            #Only run power analysis for upper right triangular matrix
            if ind2_sil > ind1_sil:
                print('\nsil_1 ' + str(sil1) + '; sil_2 ' + str(sil2))

                #Iterate through all conditions
                for cond in conditions[0:4]:
                    print('\n\tcond: ' + str(cond))

                    #Update guesses
                    if ind2_sil == ind1_sil + 1 or guess[cond] == 2048:
                        guess[cond] = np.int(2048)
                    else:
                        #Ctrl group
                        exp_2 = np.ceil(np.log(
                                minsamples[cond][ind1_sil, ind2_sil - 1])
                                / np.log(2))
                        guess[cond] = np.int(2 ** (exp_2))

                    #Calculate minsamples based on cond
                    if cond is 'binary':
                        minsamples[cond][ind1_sil, ind2_sil] = power_analysis(
                                binary_calc[ind1_sil], binary_calc[ind2_sil],
                                init_guess = guess[cond], beta = conds_beta[cond],
                                ctrl_n = conds_ctrl_n[cond],
                                sample_draws = sample_draws,
                                stat_test = 'chisquare')
                    else:
                        minsamples[cond][ind1_sil, ind2_sil] = power_analysis(
                                fra_calc[ind1_sil], fra_calc[ind2_sil],
                                init_guess = guess[cond], beta = conds_beta[cond],
                                ctrl_n = conds_ctrl_n[cond],
                                sample_draws = sample_draws)



                    guess[cond] = minsamples[cond][ind1_sil, ind2_sil]

                    #Set mirror image on other side to same value
                    minsamples[cond][ind2_sil, ind1_sil] = \
                    minsamples[cond][ind1_sil, ind2_sil]

                #Set same-same comparisons to arbitrarily high val
            elif ind2_sil == ind1_sil:
                for cond in conditions[0:4]:
                    minsamples[cond][ind1_sil, ind2_sil] = np.int(2048)
    #-----------------
    #Minsamples 2: LLR/Wilke's statistical test

    #Create likelihood function for each possible observation
    step = 0.01
    obs = np.arange(-2, 1+2*step, step) #Bin observations from -200 to 100
    likelihood = np.empty([len(obs), len(silent_truefrac)])
    for ind_obs, obs_ in enumerate(obs):
        for ind_hyp, hyp_ in enumerate(silent_truefrac):
            # obs_in_range = np.where(np.logical_and(
            #     fra_calc[ind_hyp] < obs_ + step/2, fra_calc[ind_hyp] > obs_ - step/2))[0]
            obs_in_range = np.where(np.abs(fra_calc[ind_hyp] - obs_) < step/2)[0]
            p_obs = len(obs_in_range) / len(fra_calc[ind_hyp])
            likelihood[ind_obs, ind_hyp] = p_obs
    likelihood += 0.0001 #Set likelihoods microscopically away from 0 to avoid log(0) errors
    #Minsamples for the failure-rate analysis
    for ind_sil, sil in enumerate(silent_truefrac):
        print('\nsil ' + str(sil))
        #Iterate through all conditions
        for cond in conditions[4:6]:
            if cond is not 'llr_binary':
                print('\tcond ' + str(cond))
                #Update guesses
                if guess[cond] == 2048:
                    guess[cond] = np.int(2048)
                else:
                    #Ctrl group
                    exp_2 = np.ceil(np.log(
                            minsamples[cond][ind_sil - 1])
                            / np.log(2))
                    guess[cond] = np.int(2 ** (exp_2))

                #Calculate minsamples
                minsamples[cond][ind_sil] = power_analysis_llr(
                        fra_calc[ind_sil], likelihood,
                        init_guess = guess[cond], beta = conds_beta[cond],
                        sample_draws = sample_draws)

                guess[cond] = minsamples[cond][ind_sil]

    #Lastly, update binary condition with analytical solution
    silent_truefrac_nozeroes = silent_truefrac + 1/10000 #Move away from 0 to avoid log0 errors
    minsamples['llr_binary'] = np.log(conds_beta['llr_binary']) / (np.log(1-silent_truefrac_nozeroes))


    #-----------------
    #Calc discriminability

    #Initialize vars
    discrim_dsilent = silent_truefrac

    discrim_nsamples_templ = np.empty_like(silent_truefrac, dtype = np.ndarray)
    discrim_nsamp_mean_templ = np.empty_like(silent_truefrac)
    discrim_nsamp_std_templ = np.empty_like(silent_truefrac)

    discrim_nsamples = {cond: discrim_nsamples_templ.copy() for cond in conditions}
    discrim_nsamp_mean = {cond: discrim_nsamp_mean_templ.copy() for cond in conditions}
    discrim_nsamp_std = {cond: discrim_nsamp_std_templ.copy() for cond in conditions}

    #Store coords of each discrim value
    discrim_xax_coords = {cond: discrim_nsamples_templ.copy() for cond in conditions}
    discrim_yax_coords = {cond: discrim_nsamples_templ.copy() for cond in conditions}
    discrim_zax_coords = {cond: discrim_nsamples_templ.copy() for cond in conditions}

    #Xaxis and Yaxis storages for 3d plots
    xax_3d = np.empty_like(mins_templ)
    yax_3d = np.empty_like(mins_templ)

    #Iterate through all combos of silents and store in discrim_nsamples
    for ind1_sil, sil1 in enumerate(silent_truefrac):
        for ind2_sil , sil2 in enumerate(silent_truefrac):

            #Store x and y coords for this combo of silents
            xax_3d[ind1_sil, ind2_sil] = sil1
            yax_3d[ind1_sil, ind2_sil] = sil2

            for cond in conditions[0:4]:
                #Calculate difference in silents and find corresp val in discrim_dsilent
                diff_silent_ = np.abs(sil1 - sil2)
                ind_discrim_ = np.argmin(np.abs(discrim_dsilent
                                               - diff_silent_)).astype(np.int)

                #store the number of samples for this (ind1, ind2) pair in the
                #appropriate slot of discrim_nsamples[cond] by appending
                if discrim_nsamples[cond][ind_discrim_] is None:
                    discrim_nsamples[cond][ind_discrim_] = np.array(
                            minsamples[cond][ind1_sil, ind2_sil])

                    #Store x,y,z coords on graph if upper right triangular
                    if ind2_sil > ind1_sil:
                        discrim_xax_coords[cond][ind_discrim_] = sil1
                        discrim_yax_coords[cond][ind_discrim_] = sil2
                        discrim_zax_coords[cond][ind_discrim_] = np.array(
                            minsamples[cond][ind1_sil, ind2_sil])

                else:
                    discrim_nsamples[cond][ind_discrim_] = np.append(
                            discrim_nsamples[cond][ind_discrim_],
                            minsamples[cond][ind1_sil, ind2_sil])

                    if ind2_sil > ind1_sil:
                        discrim_xax_coords[cond][ind_discrim_] = np.append(
                            discrim_xax_coords[cond][ind_discrim_], sil1)
                        discrim_yax_coords[cond][ind_discrim_] = np.append(
                            discrim_yax_coords[cond][ind_discrim_], sil2)
                        discrim_zax_coords[cond][ind_discrim_] = np.append(
                            discrim_zax_coords[cond][ind_discrim_], np.array(
                            minsamples[cond][ind1_sil, ind2_sil]))

    #Calculate the mean and stdev for each cond
    for ind_discrim, dsil in enumerate(discrim_dsilent):
        for cond in conditions[0:4]:
            discrim_nsamp_mean[cond][ind_discrim] = np.mean(
                    discrim_nsamples[cond][ind_discrim])
            discrim_nsamp_std[cond][ind_discrim] = np.std(
                    discrim_nsamples[cond][ind_discrim])


    ############################################################################
    #Make figure
    ############################################################################
        plt.style.use('publication_pnas_ml')

        fig = plt.figure(constrained_layout=True)
        fig.set_figheight(6.5)
        fig.set_figwidth(3.43)
        plt.rc('font', size = 6)

        #Define spec for entire fig
        spec_all = gridspec.GridSpec(nrows = 3, ncols = 2, height_ratios = [1.5, 1, 1],
                                     width_ratios = [1, 2], figure = fig)

        #Spec for top right
        spec_top = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                         subplot_spec = spec_all[0, 1],
                                                         wspace = 0.3, hspace = 0.5,
                                                         width_ratios = [2, 1])

        spec_middle = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                     subplot_spec = spec_all[1, 1],
                                                     wspace = 0.3, hspace = 0.5)
        #Spec for bottom right
        spec_bottom = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                     subplot_spec = spec_all[2, 1],
                                                     wspace = 0.4, hspace = 0.5)

        # plt.rc('svg', fonttype = 'none')
        #
        # xsmall_textsize = fontsize - 4
        # small_textsize = fontsize - 3
        # medium_textsize = fontsize - 1
        # large_textsize = fontsize
        #
        # plt.rc('font', size = small_textsize)          # controls default text sizes
        # plt.rc('axes', titlesize = medium_textsize,
        #        labelsize = small_textsize, labelpad = 2)    # fontsize of the axes title
        # plt.rc('xtick', labelsize = xsmall_textsize)    # fontsize of the tick labels
        # plt.rc('ytick', labelsize = xsmall_textsize)    # fontsize of the tick labels
        #plt.rc('legend', fontsize = fontsize - 3)    # legend fontsize

        #Define colors to be used
        color_1 = np.array([0.25, 0.55, 0.18])

        #---------------------------
        #Top right: 3d plot and histograph
        subp_3d = fig.add_subplot(spec_top[0, 0], projection='3d')
        surf = subp_3d.plot_surface(xax_3d * 100, yax_3d * 100,
                                    minsamples['base'].clip(max = 100, min = 0),
                                    rstride = 1, cstride = 1, vmax = 100, vmin = 0,
                                    cmap = cm.coolwarm, alpha = 0.5)
        subp_3d.set_zlim(0, 100)
        subp_3d.set_xlabel('true silent (%)')
        subp_3d.set_ylabel('true silent (%)')
        subp_3d.set_zlabel('min samples')
        # subp_3d.tick_params(labelsize = small_textsize)
        subp_3d.view_init(elev = 30, azim = 70)
        subp_3d.set_title('Power analysis', alpha = 0.5, fontweight = 'bold',
                                loc = 'left')

        #Histogram subplot: First, calculate where discrim is 0.1, 0.2, 0.3
        discrims = [0.1, 0.2, 0.3]
    #    color_d = [[0, 0.2, 0.05], [0, 0.4, 0.1], [0, 0.6, 0.2]]
        color_d = [[0, 0.2, 0.1], [0, 0.4, 0.2], [0, 0.6, 0.3]]
        alpha_d = [0.65, 0.5, 0.35]

        subp_3d_extra = fig.add_subplot(spec_top[0, 1])

        for ind, discrim in enumerate(discrims):
            #Calc index in x,y,z coords where discrim is true
            ind_disc = np.argmin(np.abs(discrim_dsilent
                                   - discrim)).astype(np.int)

            #Plot lines on subp_3d
            subp_3d.plot(discrim_xax_coords['base'][ind_disc] * 100,
                         discrim_yax_coords['base'][ind_disc] * 100,
                         np.clip(discrim_zax_coords['base'][ind_disc], 0, 100),
                         color = color_d[ind], alpha = alpha_d[ind],
                         linewidth = 3)

            #Plot histograms on subp_3d_extra
            subp_3d_extra.hist(discrim_nsamples['base'][ind_disc],
                               bins = np.logspace(np.log10(10),np.log10(1000), 25),
                               weights = (np.ones_like(
                                       discrim_nsamples['base'][ind_disc])
                                       / len(discrim_nsamples['base'][ind_disc])),
                               color = color_d[ind], alpha = alpha_d[ind],
                               histtype='step', linewidth = 2)

        subp_3d_extra.legend(labels = ['10', '20', '30'], title = '$\Delta$ silent (%)', frameon = False)
        subp_3d_extra.spines['right'].set_visible(False)
        subp_3d_extra.spines['top'].set_visible(False)
        subp_3d_extra.yaxis.set_ticks_position('left')
        subp_3d_extra.xaxis.set_ticks_position('bottom')
        subp_3d_extra.set_ylim([0, 0.6])
        subp_3d_extra.set_xscale("log")
        subp_3d_extra.set_xlim([0, 10 ** 3])
        subp_3d_extra.set_yticks([0, 0.3, 0.6])
        subp_3d_extra.set_xlabel('minimum samples')
        subp_3d_extra.set_ylabel('probability density')
        subp_3d_extra.set_title('Discriminability \ndistributions', alpha = 0.5, fontweight = 'bold',
                                loc = 'left')

        #---------------------------
        #Middle right: Power analysis curves

        #First curve: basic power analysis
        subp_base = fig.add_subplot(spec_middle[0, 0])

        subp_base.plot(discrim_dsilent * 100, discrim_nsamp_mean['base'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        subp_base.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['base']
                                 + discrim_nsamp_std['base'], discrim_nsamp_mean['base']
                                 - discrim_nsamp_std['base'], facecolor = color_1,
                                 alpha = 0.1)
        subp_base.spines['right'].set_visible(False)
        subp_base.spines['top'].set_visible(False)
        subp_base.yaxis.set_ticks_position('left')
        subp_base.xaxis.set_ticks_position('bottom')
        subp_base.set_xlim([0, 50])
        subp_base.set_ylim([0, 1024])
        subp_base.set_xticks([0, 10, 20, 30, 40, 50])
        subp_base.set_yticks([0, 256, 512, 768, 1024])
        subp_base.set_xlabel('Detectable $\Delta$ silent (%)')
        subp_base.set_ylabel('minimum samples required')
        subp_base.set_title('Discriminability',
                              alpha = 0.5, fontweight = 'bold', loc = 'left')

        subp_base_inset = inset_axes(subp_base, width='60%', height='70%', loc = 1)
        subp_base_inset.plot(discrim_dsilent * 100, discrim_nsamp_mean['base'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        subp_base_inset.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['base']
                                 + discrim_nsamp_std['base'], discrim_nsamp_mean['base']
                                 - discrim_nsamp_std['base'], facecolor = color_1,
                                 alpha = 0.1)
        subp_base_inset.spines['right'].set_visible(False)
        subp_base_inset.spines['top'].set_visible(False)
        subp_base_inset.yaxis.set_ticks_position('left')
        subp_base_inset.xaxis.set_ticks_position('bottom')
        subp_base_inset.set_xlim([15, 50])
        subp_base_inset.set_ylim([0, 50])
        subp_base_inset.set_xticks([15, 30, 45])
        subp_base_inset.set_yticks([0, 10, 20, 30, 40, 50])
        # subp_base_inset.tick_params(labelsize = xsmall_textsize)
        mark_inset(subp_base, subp_base_inset, loc1=3, loc2=4, fc="none",
                   ec="0.5", ls = ':', lw = 2)


        #-----------------------------------------------
        #Second curve: power analysis with changes in control group
        subp_ctrln = fig.add_subplot(spec_middle[0, 1])

        color_n10 = [0.5, 0.2, 0.3]
        color_n20 = [0.8, 0.1, 0.2]

        base_ = subp_ctrln.plot(discrim_dsilent * 100, discrim_nsamp_mean['base'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        subp_ctrln.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['base']
                                 + discrim_nsamp_std['base'], discrim_nsamp_mean['base']
                                 - discrim_nsamp_std['base'], facecolor = color_1,
                                 alpha = 0.1)

        n10 = subp_ctrln.plot(discrim_dsilent * 100, discrim_nsamp_mean['c10'],
                      color = color_n10, alpha = 0.6, linewidth = 2)
        subp_ctrln.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['c10']
                                 + discrim_nsamp_std['c10'], discrim_nsamp_mean['c10']
                                 - discrim_nsamp_std['c10'], facecolor = color_n10,
                                 alpha = 0.1)

        n20 = subp_ctrln.plot(discrim_dsilent * 100, discrim_nsamp_mean['c20'],
                      color = color_n20, alpha = 0.6, linewidth = 2)
        subp_ctrln.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['c20']
                                 + discrim_nsamp_std['c20'], discrim_nsamp_mean['c20']
                                 - discrim_nsamp_std['c20'], facecolor = color_n20,
                                 alpha = 0.1)
        subp_ctrln.legend([base_[-1], n10[-1], n20[-1]], ['exp n', '10', '20'], title = 'control n',
                          frameon = False, loc = 1)
        subp_ctrln.spines['right'].set_visible(False)
        subp_ctrln.spines['top'].set_visible(False)
        subp_ctrln.yaxis.set_ticks_position('left')
        subp_ctrln.xaxis.set_ticks_position('bottom')
        subp_ctrln.set_xlim([0, 50])
        subp_ctrln.set_ylim([0, 1024])
        subp_ctrln.set_xticks([0, 10, 20, 30, 40, 50])
        subp_ctrln.set_yticks([0, 256, 512, 768, 1024])
        subp_ctrln.set_xlabel('Detectable $\Delta$ silent (%)')
        subp_ctrln.set_ylabel('minimum samples required')
        subp_ctrln.set_title('Discriminability: \n$\Delta$ ctrl n',
                              alpha = 0.5, fontweight = 'bold', loc = 'left')

        #-----------------------------------------------
        #Third curve: power analysis with binary discriminability
        subp_binary = fig.add_subplot(spec_bottom[0, 0])

        color_binary = [0.2, 0.2, 0.2]

        base_ = subp_binary.plot(discrim_dsilent * 100, discrim_nsamp_mean['base'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        subp_binary.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['base']
                                 + discrim_nsamp_std['base'], discrim_nsamp_mean['base']
                                 - discrim_nsamp_std['base'], facecolor = color_1,
                                 alpha = 0.1)

        binary = subp_binary.plot(discrim_dsilent * 100, discrim_nsamp_mean['binary'],
                      color = color_binary, alpha = 0.6, linewidth = 2)
        subp_binary.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['binary']
                                 + discrim_nsamp_std['binary'], discrim_nsamp_mean['binary']
                                 - discrim_nsamp_std['binary'], facecolor = color_binary,
                                 alpha = 0.1)
        subp_binary.spines['right'].set_visible(False)
        subp_binary.spines['top'].set_visible(False)
        subp_binary.yaxis.set_ticks_position('left')
        subp_binary.xaxis.set_ticks_position('bottom')
        subp_binary.set_xlim([0, 50])
        subp_binary.set_ylim([0, 1024])
        subp_binary.set_xticks([0, 10, 20, 30, 40, 50])
        subp_binary.set_yticks([0, 256, 512, 768, 1024])
        subp_binary.set_xlabel('Detectable $\Delta$ silent (%)')
        subp_binary.set_ylabel('minimum samples required')
        subp_binary.set_title('Discriminability: \n$\Delta$ ctrl n',
                              alpha = 0.5, fontweight = 'bold', loc = 'left')

        subp_binary_inset = inset_axes(subp_binary, width='60%', height='70%', loc = 1)
        base_ = subp_binary_inset.plot(discrim_dsilent * 100, discrim_nsamp_mean['base'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        subp_binary_inset.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['base']
                                 + discrim_nsamp_std['base'], discrim_nsamp_mean['base']
                                 - discrim_nsamp_std['base'], facecolor = color_1,
                                 alpha = 0.1)
        b02_ = subp_binary_inset.plot(discrim_dsilent * 100, discrim_nsamp_mean['binary'],
                      color = color_binary, alpha = 0.6, linewidth = 2)
        subp_binary_inset.fill_between(discrim_dsilent * 100, discrim_nsamp_mean['binary']
                                 + discrim_nsamp_std['binary'], discrim_nsamp_mean['binary']
                                 - discrim_nsamp_std['binary'], facecolor = color_binary,
                                 alpha = 0.1)
        subp_binary_inset.legend([base_[-1], binary[-1]], ['FRA', 'binary'],
                          frameon = False, loc = 1)
        subp_binary_inset.spines['right'].set_visible(False)
        subp_binary_inset.spines['top'].set_visible(False)
        subp_binary_inset.yaxis.set_ticks_position('left')
        subp_binary_inset.xaxis.set_ticks_position('bottom')
        subp_binary_inset.set_xlim([15, 50])
        subp_binary_inset.set_ylim([0, 200])
        subp_binary_inset.set_xticks([15, 30, 45])
        subp_binary_inset.set_yticks([0, 50, 100, 150, 200])
        # subp_binary_inset.tick_params(labelsize = xsmall_textsize)
        mark_inset(subp_binary, subp_binary_inset, loc1=3, loc2=4, fc="none",
                   ec="0.5", ls = ':', lw = 2)

        #-----------------------------------------------
        #Fourth curve: power analysis with binary discriminability
        subp_llr = fig.add_subplot(spec_bottom[0, 1])

        color_llr = [0.2, 0.2, 0.2]

        llr_base = subp_llr.plot(silent_truefrac * 100, minsamples['llr'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        llr_binary = subp_llr.plot(silent_truefrac * 100, minsamples['llr_binary'],
                      color = color_binary, alpha = 0.6, linewidth = 2)
        subp_llr.spines['right'].set_visible(False)
        subp_llr.spines['top'].set_visible(False)
        subp_llr.yaxis.set_ticks_position('left')
        subp_llr.xaxis.set_ticks_position('bottom')
        subp_llr.set_xlim([0, 50])
        subp_llr.set_ylim([0, 128])
        subp_llr.set_xticks([0, 10, 20, 30, 40, 50])
        subp_llr.set_yticks([0, 32, 64, 96, 128])
        subp_llr.set_xlabel('silent synapses (%)')
        subp_llr.set_ylabel('minimum samples required')
        subp_llr.set_title('Log-likelihood \nratio vs null',
                              alpha = 0.5, fontweight = 'bold', loc = 'left')

        subp_llr_inset = inset_axes(subp_llr, width='60%', height='70%', loc = 1)
        llr_base_inset = subp_llr_inset.plot(silent_truefrac * 100, minsamples['llr'], color = color_1,
                           alpha = 0.6, linewidth = 2)
        llr_binary_inset = subp_llr_inset.plot(silent_truefrac * 100, minsamples['llr_binary'],
                      color = color_binary, alpha = 0.6, linewidth = 2)


        subp_llr_inset.spines['right'].set_visible(False)
        subp_llr_inset.spines['top'].set_visible(False)
        subp_llr_inset.yaxis.set_ticks_position('left')
        subp_llr_inset.xaxis.set_ticks_position('bottom')
        subp_llr_inset.set_xlim([15, 50])
        subp_llr_inset.set_ylim([0, 10])
        subp_llr_inset.set_xticks([15, 30, 45])
        subp_llr_inset.set_yticks([0, 5, 10])
        subp_llr_inset.legend([llr_base_inset, llr_binary_inset], ['LLR-FRA', 'LLR-binary'],
                        frameon = False)
        # subp_llr_inset.tick_params(labelsize = xsmall_textsize)
        mark_inset(subp_llr, subp_llr_inset, loc1=3, loc2=4, fc="none",
            ec="0.5", ls = ':', lw = 2)


        #---------------------------------
        #Set tight layout and save
        fig.set_constrained_layout_pads(w_pad=0.001, h_pad=0.001,
                hspace=0.01, wspace=0.01)
    #    spec_all.tight_layout(fig)
        plt.savefig(figname, bbox_inches = 'tight')

    return

def plot_fig4_suppLLR(n_true_silents = 100, fontsize = 12, sample_draws = 5000,
              figname = 'Fig4_LLR.pdf'):
    '''
    Plot the power analysis figure: Compare null to experimental cases
    - 1a. Use log-likelihood ratio to analytically compute power analysis curve for
        binary comparisons
    - 1b. Use numerical simulations to compute power analysis curve from FRA methods
    - 2/3: Redo 1, but with changes in control sample sizes?

    '''
    n_true_silents = 50
    fontsize = 12
    sample_draws = 5000
    figname = 'Fig4_supp.pdf'

    ##################
    #1. Simulations
    ##################

    silent_truefrac = np.linspace(0, 0.5, num = n_true_silents)
    fra_calc = np.empty(len(silent_truefrac), dtype = np.ndarray)

    #Set up multiple conditions
    conditions = ['base', 'binary']

    #Set parameters for each condition
    conds_beta = {cond: 0.2 for cond in conditions}
    conds_ctrl_n = {cond: False for cond in conditions}

    mins_templ = np.empty(len(silent_truefrac))
    minsamples = {cond : mins_templ.copy() for cond in conditions}

    print('Generating FRA calcs...')
    #Generate FRA calcs and calc simple minsamples versus baseline
    for ind, silent in enumerate(silent_truefrac):
        print('\tsilent: ' + str(silent))
        #Generate an FRA dist
        fra_calc[ind] = gen_fra_dist(n_simulations = 10000,
                                             silent_fraction = silent,
                                             zeroing = True,
                                             unitary_reduction = False,
                                             frac_reduction = 0.2)


    #Create loglikelihood function for each case
    step = 0.01
    obs = np.arange(-2, 1+2*step, step) #Bin observations from -200 to 100
    likelihood = np.empty([len(obs), len(silent_truefrac)])
    for ind_obs, obs_ in enumerate(obs):
        for ind_hyp, hyp_ in enumerate(silent_truefrac):
            # obs_in_range = np.where(np.logical_and(
            #     fra_calc[ind_hyp] < obs_ + step/2, fra_calc[ind_hyp] > obs_ - step/2))[0]
            obs_in_range = np.where(np.abs(fra_calc[ind_hyp] - obs_) < step/2)[0]
            p_obs = len(obs_in_range) / len(fra_calc[ind_hyp])
            likelihood[ind_obs, ind_hyp] = p_obs
    likelihood += 0.0001 #Set likelihoods microscopically away from 0 to avoid log(0) errors

    #------------------
    #Set up guesses for each condition to start at
    guess = {cond : int(2048) for cond in conditions}

    #Minsamples for the failure-rate analysis
    for ind_sil, sil in enumerate(silent_truefrac):
        print('\nsil ' + str(sil))
        #Iterate through all conditions
        for cond in conditions:
            if cond is not 'binary':
                print('\tcond ' + str(cond))
                #Update guesses
                if guess[cond] == 2048:
                    guess[cond] = np.int(2048)
                else:
                    #Ctrl group
                    exp_2 = np.ceil(np.log(
                            minsamples[cond][ind_sil - 1])
                            / np.log(2))
                    guess[cond] = np.int(2 ** (exp_2))

                #Calculate minsamples
                minsamples[cond][ind_sil] = power_analysis_llr(
                        fra_calc[ind_sil], likelihood,
                        init_guess = guess[cond], beta = conds_beta[cond],
                        sample_draws = sample_draws)

                guess[cond] = minsamples[cond][ind_sil]

    #Lastly, update binary condition with analytical solution
    silent_truefrac_nozeroes = silent_truefrac + 1/10000 #Move away from 0 to avoid log0 errors
    minsamples['binary'] = np.log(conds_beta['binary']) / (np.log(1-silent_truefrac_nozeroes))
    ##
    # inds = np.random.choice(np.arange(230, 250), 100)≈≈Ω≈ç
    # likelihood_sum = np.sum(np.log(likelihood[inds, :]), axis = 0)
    # plt.figure(); plt.plot(likelihood_sum)
    ##
    ##############
    #Make figs
    ##############
    fig = plt.figure()

    fig.set_figheight(8)
    fig.set_figwidth(8)

    #Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows = 3, ncols = 2, height_ratios = [1.5, 1, 1],
                                 width_ratios = [1, 2])

    #Spec for top right
    spec_top = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                     subplot_spec = spec_all[0, 1],
                                                     wspace = 0.3, hspace = 0.5,
                                                     width_ratios = [2, 1])

    spec_middle = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec = spec_all[1, 1],
                                                 wspace = 0.3, hspace = 0.5)
    #Spec for bottom right
    spec_bottom = gridspec.GridSpecFromSubplotSpec(1, 2,
                                                 subplot_spec = spec_all[2, 1],
                                                 wspace = 0.4, hspace = 0.5)

    plt.rc('svg', fonttype = 'none')

    xsmall_textsize = fontsize - 4
    small_textsize = fontsize - 3
    medium_textsize = fontsize - 1
    large_textsize = fontsize

    plt.rc('font', size = small_textsize)          # controls default text sizes
    plt.rc('axes', titlesize = medium_textsize,
           labelsize = small_textsize, labelpad = 2)    # fontsize of the axes title
    plt.rc('xtick', labelsize = xsmall_textsize)    # fontsize of the tick labels
    plt.rc('ytick', labelsize = xsmall_textsize)    # fontsize of the tick labels
    plt.rc('legend', fontsize = xsmall_textsize)    # legend fontsize
    plt.rc('figure', titlesize = large_textsize)

    #Define colors to be used
    color_1 = np.array([0.25, 0.55, 0.18])


    #First curve: basic power analysis
    subp_base = fig.add_subplot(spec_middle[0, 0])

    subp_base.plot(silent_truefrac * 100, minsamples['base'], color = color_1,
                       alpha = 0.6, linewidth = 2)
    subp_base.spines['right'].set_visible(False)
    subp_base.spines['top'].set_visible(False)
    subp_base.yaxis.set_ticks_position('left')
    subp_base.xaxis.set_ticks_position('bottom')
    subp_base.set_xlim([0, 50])
    subp_base.set_ylim([0, 1024])
    subp_base.set_xticks([0, 10, 20, 30, 40, 50])
    subp_base.set_yticks([0, 256, 512, 768, 1024])
    subp_base.set_xlabel('silent synapses (%)')
    subp_base.set_ylabel('minimum samples required')
    subp_base.set_title('FRA power analysis',
                          alpha = 0.5, fontweight = 'bold', loc = 'left')

    subp_base_inset = inset_axes(subp_base, width='60%', height='70%', loc = 1)
    subp_base_inset.plot(silent_truefrac * 100, minsamples['base'], color = color_1,
                       alpha = 0.6, linewidth = 2)
    subp_base_inset.spines['right'].set_visible(False)
    subp_base_inset.spines['top'].set_visible(False)
    subp_base_inset.yaxis.set_ticks_position('left')
    subp_base_inset.xaxis.set_ticks_position('bottom')
    subp_base_inset.set_xlim([15, 50])
    subp_base_inset.set_ylim([0, 50])
    subp_base_inset.set_xticks([15, 30, 45])
    subp_base_inset.set_yticks([0, 10, 20, 30, 40, 50])
    subp_base_inset.tick_params(labelsize = xsmall_textsize)
    mark_inset(subp_base, subp_base_inset, loc1=3, loc2=4, fc="none",
               ec="0.5", ls = ':', lw = 2)

    #-----------------------------------------------
    #Fourth curve: power analysis with binary discriminability
    subp_binary = fig.add_subplot(spec_bottom[0, 1])

    color_binary = [0.2, 0.2, 0.2]

    base_ = subp_binary.plot(silent_truefrac * 100, minsamples['base'], color = color_1,
                       alpha = 0.6, linewidth = 2)
    binary = subp_binary.plot(silent_truefrac * 100, minsamples['binary'],
                  color = color_binary, alpha = 0.6, linewidth = 2)
    subp_binary.legend([base_[-1], binary[-1]], ['failure-rate', 'binary class.'], title = 'technique',
                      fontsize = fontsize - 4, frameon = False, loc = 1)
    subp_binary.spines['right'].set_visible(False)
    subp_binary.spines['top'].set_visible(False)
    subp_binary.yaxis.set_ticks_position('left')
    subp_binary.xaxis.set_ticks_position('bottom')
    subp_binary.set_xlim([0, 50])
    subp_binary.set_ylim([0, 1024])
    subp_binary.set_xticks([0, 10, 20, 30, 40, 50])
    subp_binary.set_yticks([0, 256, 512, 768, 1024])
    subp_binary.set_xlabel('silent synapses')
    subp_binary.set_ylabel('minimum samples required')
    subp_binary.set_title('FRA vs binary power analysiss',
                          alpha = 0.5, fontweight = 'bold', loc = 'left')

    subp_binary_inset = inset_axes(subp_binary, width='60%', height='70%', loc = 1)
    base_ = subp_binary_inset.plot(silent_truefrac * 100, minsamples['base'], color = color_1,
                       alpha = 0.6, linewidth = 2)
    b02_ = subp_binary_inset.plot(silent_truefrac * 100, minsamples['binary'],
                  color = color_binary, alpha = 0.6, linewidth = 2)
    subp_binary_inset.legend([base_[-1], binary[-1]], ['failure-rate', 'binary class.'], title = 'technique',
                      fontsize = fontsize - 4, frameon = False, loc = 1)
    subp_binary_inset.spines['right'].set_visible(False)
    subp_binary_inset.spines['top'].set_visible(False)
    subp_binary_inset.yaxis.set_ticks_position('left')
    subp_binary_inset.xaxis.set_ticks_position('bottom')
    subp_binary_inset.set_xlim([15, 50])
    subp_binary_inset.set_ylim([0, 200])
    subp_binary_inset.set_xticks([15, 30, 45])
    subp_binary_inset.set_yticks([0, 50, 100, 150, 200])
    subp_binary_inset.tick_params(labelsize = xsmall_textsize)
    mark_inset(subp_binary, subp_binary_inset, loc1=3, loc2=4, fc="none",
               ec="0.5", ls = ':', lw = 2)


    #---------------------------------
    #Set tight layout and save
    spec_all.tight_layout(fig)
    plt.savefig(figname, bbox_inches = 'tight')
    plt.show()

    return


def _gen_fra_dist_fails(method = 'iterative', pr_dist = 'uniform', silent_fraction = 0.5,
                                num_trials = 50, n_simulations = 10000, n_start = 100,
                                zeroing = False, graph_ex = False, verbose = False,
                                unitary_reduction = False, frac_reduction = 0.2,
                                binary_vals = False, failrate_low = 0.2,
                                failrate_high = 0.8):
    '''
    In-progress function to attempt to perform MLE using two variables per
    experiment: Fh and Fd. Here, the numerical simulations are performed and
    an estimate distribution along with failure rate distributions are returned.
    '''

    if verbose is True:
        print ('Generating FRA distribution with p(silent) = ', str(silent_fraction), ':')

    if binary_vals is True:
        fra_calc = np.zeros(n_simulations)
        fra_calc[0:int(silent_fraction * n_simulations)] = 1

        return fra_calc

    #First, generate realistic groups of neurons
    nonsilent_syn_group, silent_syn_group, \
    pr_nonsilent_syn_group, pr_silent_syn_group \
    = draw_subsample(method = method, pr_dist = pr_dist,
                                       n_simulations = n_simulations,
                                       n_start = n_start,
                                       silent_fraction = silent_fraction,
                                       failrate_low = failrate_low,
                                       failrate_high = failrate_high,
                                       unitary_reduction = unitary_reduction,
                                       frac_reduction = frac_reduction,
                                       verbose = verbose)

    #Calculate p(failure) mathematically for hyperpol, depol based on product of (1- pr)
    math_failure_rate_hyperpol = np.ma.prod(1 - pr_nonsilent_syn_group, axis = 1).compressed()

    pr_all_depol = np.ma.append(pr_nonsilent_syn_group, pr_silent_syn_group, axis = 1)
    math_failure_rate_depol = np.ma.prod(1 - pr_all_depol, axis = 1).compressed()

    #Simulate trials where failure rate is binary, calculate fraction fails
    sim_failure_rate_hyperpol = np.sum(np.random.rand(n_simulations, num_trials) < np.tile(
            math_failure_rate_hyperpol, (num_trials, 1)).transpose(), axis = 1) / num_trials

    sim_failure_rate_depol = np.sum(np.random.rand(n_simulations, num_trials) < np.tile(
            math_failure_rate_depol, (num_trials, 1)).transpose(), axis = 1) / num_trials

    #Calculate failure rate
    fra_calc = 1 - np.log(sim_failure_rate_hyperpol) / np.log(sim_failure_rate_depol)

    #Filter out oddities
    fra_calc[fra_calc == -(np.inf)] = 0
    fra_calc[fra_calc == np.inf] = 0
    fra_calc[fra_calc == np.nan] = 0
    fra_calc = np.nan_to_num(fra_calc)

    if zeroing is True:
        fra_calc[fra_calc < 0] = 0

    return fra_calc, sim_failure_rate_hyperpol, sim_failure_rate_depol

def _mle_fh_fd():
    '''
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

    '''
    n_simulations = 20000
    obs_bins = 0.02
    zeroing = False

    hyp = np.linspace(0, 0.5, num = 200)
    fra_calc = np.empty(len(hyp), dtype = np.ndarray)
    fra_fail_h = np.empty(len(hyp), dtype = np.ndarray)
    fra_fail_d = np.empty(len(hyp), dtype = np.ndarray)


    print('Generating FRA calcs...')
    for ind, silent in enumerate(hyp):
        print('\tSilent frac: ', str(silent))
        #Generate an FRA dist
        fra_calc[ind], fra_fail_h[ind], fra_fail_d[ind] \
            = _gen_fra_dist_fails(n_simulations = n_simulations,
            silent_fraction = silent, zeroing = zeroing)

    #2. Create loglikelihood function for each case
    obs_fh = np.arange(0, 1, obs_bins) #Bin observations from -200 to 100
    obs_fd = np.arange(0, 1, obs_bins) #Bin observations from -200 to 100


    likelihood = np.empty([len(obs_fh), len(obs_fd), len(hyp)])

    for ind_obs_fh, obs_fh_ in enumerate(obs_fh):
        for ind_obs_fd, obs_fd_ in enumerate(obs_fd):
            for ind_hyp, hyp_ in enumerate(hyp):
                obs_in_range_fh_ = np.where(np.abs(fra_fail_h[ind_hyp] - obs_fh_) < obs_bins/2)[0]
                obs_in_range_fd_ = np.where(np.abs(fra_fail_d[ind_hyp] - obs_fd_) < obs_bins/2)[0]

                obs_in_range_both = set(obs_in_range_fh_) \
                    - (set(obs_in_range_fh_) - set(obs_in_range_fd_))

                p_obs = len(obs_in_range_both) / len(fra_fail_h[ind_hyp])

                likelihood[ind_obs_fh, ind_obs_fd, ind_hyp] = p_obs
    likelihood += 0.0001 #Set likelihoods microscopically away from 0 to avoid log(0) errors

    #-----------------------------
    #Plotting
    #-----------------------------
    plt.style.use('presentation_ml')

    fig = plt.figure(figsize = (10, 6))
    spec = gridspec.GridSpec(nrows = 2, ncols = 3,
        height_ratios = [1, 0.2], width_ratios = [0.2, 1, 1])

    p_silent_1 = 0.1
    ind_psil_1 = np.abs(hyp - p_silent_1).argmin()
    p_silent_2 = 0.4
    ind_psil_2 = np.abs(hyp - p_silent_2).argmin()

    ax_failcomp = fig.add_subplot(spec[0, 1])
    ax_failcomp.plot(fra_fail_h[ind_psil_1], fra_fail_d[ind_psil_1], '.',
        color = plt.cm.RdBu(0.2), alpha = 0.1)
    ax_failcomp.plot(fra_fail_h[ind_psil_2], fra_fail_d[ind_psil_1], '.',
        color = plt.cm.RdBu(0.8), alpha = 0.1)


    ax_failcomp_fd = fig.add_subplot(spec[0, 0])
    ax_failcomp_fh = fig.add_subplot(spec[1, 1])

    ax_failcomp_fh.hist(fra_fail_h[ind_psil_1], bins = 50,
        density = True, facecolor = plt.cm.RdBu(0.2), alpha = 0.5)
    ax_failcomp_fh.hist(fra_fail_h[ind_psil_2], bins = 50,
        density = True, facecolor = plt.cm.RdBu(0.8), alpha = 0.5)

    ax_failcomp_fd.hist(fra_fail_d[ind_psil_1], orientation = 'horizontal', bins = 50,
        density = True, facecolor = plt.cm.RdBu(0.2), alpha = 0.5)
    ax_failcomp_fd.hist(fra_fail_d[ind_psil_2], orientation = 'horizontal', bins = 50,
        density = True, color = plt.cm.RdBu(0.8), alpha = 0.5)
    ax_failcomp_fh.set_xlabel('$F_{h}$')
    ax_failcomp_fh.set_ylabel('pdf')
    ax_failcomp_fd.set_ylabel('$F_{d}$')
    ax_failcomp_fd.set_xlabel('pdf')


    #---------
    ax_likelihood = fig.add_subplot(spec[:, 2])

    n_sims_ex = 20
    silent_frac_ex = 0.2

    fra_calc_ex, fra_fail_h_ex, fra_fail_d_ex \
        = _gen_fra_dist_fails(n_simulations = n_sims_ex,
        silent_fraction = silent_frac_ex, zeroing = False)

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


#%%

nt = 50
k = np.arange(0, nt)
nns = 100

p_k = np.zeros(len(k))

for ind, k_ in enumerate(k):
    p_k[ind] = sp.special.comb(int(nt), int(k_)) * (0.5**(1/nns))**(nns*k_) * (1- (0.5**(1/nns))**nns)**(nt-k_)



fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(k, p_k)
ax.set_xlabel('k')
ax.set_ylabel('p(k)')
