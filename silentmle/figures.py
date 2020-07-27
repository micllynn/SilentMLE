"""
figures.py: Functions for recreating the figures found in the publication

Author: mbfl
Date: 19.9
"""

import os
import numpy as np
import scipy as sp
import scipy.stats as sp_stats
import h5py

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

try:
    from .core import *
    from .mle import *
except ModuleNotFoundError:
    from silentmle.core import *
    from silentmle.mle import *

# Define global colormap for red_blue
cm_ = sns.diverging_palette(240, 15, l=45, s=90, center="dark", as_cmap=True)


def plot_fig1(figname='Figure1.pdf',
              fontsize=8,
              cmap=cm_,
              cmap_hyp=0.1,
              cmap_dep=0.9):

    try:
        plt.style.use('publication_ml')
    except FileNotFoundError:
        pass

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(3.43)
    fig.set_figwidth(3.43)
    plt.rc('font', size=fontsize)

    # Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows=4, ncols=2, figure=fig,
                                 height_ratios=[1, 0.5, 0.5, 1])

    # Define bins for all hists
    bins_ = np.arange(-150, 50, 8)

    ##########################
    # Plot example simulation
    ########################

    sim_ex = fig.add_subplot(spec_all[2, 1])

    currents = np.where(np.random.rand((50)) < 0.5, 0.0, -10.0)
    currents = np.append(currents,
                         np.where(np.random.rand((50)) < 0.5, 0.0, 10.0))
    currents += (np.random.rand(len(currents)) - 0.5) * 2

    suc_fh_ind = np.where(np.abs(currents[0:50]) > 5)[0]
    suc_fd_ind = np.where(np.abs(currents[50:]) > 5)[0] + 50
    fail_ind = np.where(np.abs(currents) < 5)[0]

    sim_ex.plot(suc_fh_ind, currents[suc_fh_ind], '.',
                color=cmap(cmap_hyp))
    sim_ex.plot(suc_fd_ind, currents[suc_fd_ind], '.',
                color=cmap(cmap_dep))
    sim_ex.plot(fail_ind, currents[fail_ind], '.',
                color=[0.7, 0.7, 0.7])
    sim_ex.set_xlim([0, 100])

    sim_ex.set_ylim([-15, 15])
    sim_ex.set_xticks([])
    sim_ex.set_yticks([])

    failrate_hyp = np.sum(currents[0:50] > -5) / 50
    failrate_dep = np.sum(currents[50:] < 5) / 50

    txt_hyp = f'Fh = {failrate_hyp}'
    txt_dep = f'Fd = {failrate_dep}'
    estimate = (1 - np.log(failrate_hyp) / np.log(failrate_dep)) * 100
    txt_calc = f'Est. silent = {estimate:.1f} %'

    sim_ex.text(0.1,
                0.35,
                txt_hyp,
                fontsize=fontsize - 2,
                color=cmap(cmap_hyp),
                transform=sim_ex.transAxes,
                horizontalalignment='left',
                verticalalignment='bottom')
    sim_ex.text(0.9,
                0.65,
                txt_dep,
                fontsize=fontsize - 2,
                color=cmap(cmap_dep),
                transform=sim_ex.transAxes,
                horizontalalignment='right',
                verticalalignment='top')
    sim_ex.text(0.5,
                0.95,
                txt_calc,
                fontsize=fontsize - 2,
                fontweight='bold',
                color=[0, 0, 0],
                transform=sim_ex.transAxes,
                horizontalalignment='center')
    sns.despine(ax=sim_ex, top=True, right=True,
                left=True, bottom=True)

    ########################
    # Plot dist. estimates where sweep num is changing
    ########################
    sweep_change = fig.add_subplot(spec_all[3, 0])
    sw_ch_inset = inset_axes(sweep_change,
                             width='100%',
                             height='35%',
                             loc=3,
                             bbox_to_anchor=(.17, .7, .5, .95),
                             bbox_transform=sweep_change.transAxes,
                             borderpad=0)
    sw_ch_inset.patch.set_alpha(0)

    trials = [50, 100, 200, 500]

    calc = np.empty(len(trials), dtype=np.ndarray)
    calc_sd = np.empty(len(trials))

    for ind, trial in enumerate(trials):
        fails_hyp = np.sum(np.random.rand(trial, 10000) < 0.5, axis=0) / trial
        fails_dep = np.sum(np.random.rand(trial, 10000) < 0.5, axis=0) / trial
        calc[ind] = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100
        calc_sd[ind] = np.std(calc[ind])

        ratio_ = ind / (len(trials) - 1)
        red_ = 1 / (1 + np.exp(-5 * (ratio_ - 0.5)))
        color_ = [red_, 0.2, 1 - red_]

        sweep_change.hist(calc[ind],
                          bins=bins_,
                          weights=np.ones_like(calc[ind]) / len(calc[ind]),
                          color=cmap(ratio_),
                          alpha=0.8,
                          histtype='step')

        sw_ch_inset.plot(trial,
                         calc_sd[ind],
                         color=cmap(ratio_),
                         alpha=1,
                         marker='o',
                         fillstyle='full',
                         markersize=4)

    sw_ch_inset.plot(trials, calc_sd, color=[0, 0, 0],
                     linewidth=0.8)

    ylim_max = 0.4

    sweep_change.set_xlabel('Estimated silent (%)')
    sweep_change.set_xlim([-150, 60])
    sweep_change.set_xticks([-150, -100, -50, 0, 50])
    sweep_change.set_ylabel('pdf')
    sweep_change.set_ylim([0, 0.5])
    sweep_change.set_yticks(np.arange(0, 0.51, 0.1))
    sns.despine(ax=sweep_change, offset=3)

    sw_ch_inset.set_xlabel('Sweep num.', fontsize=fontsize - 2)
    sw_ch_inset.set_xlim([0, 600])
    sw_ch_inset.set_xticks([50, 200, 500])
    sw_ch_inset.set_ylabel('std. dev.', fontsize=fontsize - 2)
    sw_ch_inset.set_ylim([0, 40])
    sw_ch_inset.set_yticks(np.linspace(0, 40, 3))
    sw_ch_inset.tick_params(axis='both', which='major', labelsize=fontsize - 3)

    ########################
    # Plot dist. estimates where synapse num. is changing
    ########################
    _ylim_max = 0.2

    num_change = fig.add_subplot(spec_all[3, 1])
    n_ch_inset = inset_axes(num_change,
                            width='100%',
                            height='35%',
                            loc=3,
                            bbox_to_anchor=(.17, .7, .5, .95),
                            bbox_transform=num_change.transAxes,
                            borderpad=0)
    n_ch_inset.patch.set_alpha(0)

    n_syn = np.arange(1, 7)

    calc = np.empty(len(n_syn), dtype=np.ndarray)
    calc_sd = np.empty(len(n_syn))

    for ind, n in enumerate(n_syn):
        pr = 1 - (0.5**(1 / n))

        fails_hyp = np.sum(np.random.rand(50, 10000) <
                           (1 - pr)**n, axis=0) / 50
        fails_dep = np.sum(np.random.rand(50, 10000) <
                           (1 - pr)**n, axis=0) / 50
        calc[ind] = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100
        calc_sd[ind] = np.std(calc[ind])

        ratio_ = ind / (len(n_syn) - 1)
        red_ = 1 / (1 + np.exp(-6 * (ratio_ - 0.5)))
        color_ = [red_, 0.1, 1 - red_]

        num_change.hist(calc[ind],
                        bins=bins_,
                        weights=np.ones_like(calc[ind]) / len(calc[ind]),
                        color=cm_(ratio_),
                        alpha=0.8,
                        histtype='step')

        n_ch_inset.plot(n,
                        calc_sd[ind],
                        color=cm_(ratio_),
                        alpha=1,
                        marker='o',
                        fillstyle='full',
                        markersize=4)

    n_ch_inset.plot(n_syn, calc_sd, color=[0, 0, 0],
                    linewidth=0.8)

    num_change.set_xlabel('Estimated silent (%)')
    num_change.set_xlim([-150, 60])
    num_change.set_xticks([-150, -100, -50, 0, 50])
    num_change.set_ylabel('pdf')
    num_change.set_ylim([0, _ylim_max])
    num_change.set_yticks(np.linspace(0, _ylim_max, 5))
    sns.despine(ax=num_change, offset=3)

    n_ch_inset.set_xlabel('n synapses', fontsize=fontsize - 2)
    n_ch_inset.set_xlim([1, 7])
    n_ch_inset.set_xticks([2, 4, 6])
    n_ch_inset.set_ylabel('std. dev.', fontsize=fontsize - 2)
    n_ch_inset.set_ylim([20, 40])
    n_ch_inset.set_yticks(np.linspace(20, 40, 3))
    n_ch_inset.tick_params(axis='both', which='major', labelsize=fontsize - 3)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    fig.savefig(path_f)

    return


def plot_figS1(figname='FigureS1.pdf',
               fontsize=8,
               cmap_n=sns.cubehelix_palette(
                   start=0.4, rot=-0.6, light=0.8,
                   dark=0.2, as_cmap=True),
               bins_=np.arange(-150, 50, 10)):

    try:
        plt.style.use('publication_ml')
    except FileNotFoundError:
        pass

    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize+1
    mpl.rcParams['axes.linewidth'] = 1
    mpl.rcParams['lines.linewidth'] = 1
    mpl.rcParams['xtick.major.width'] = 1
    mpl.rcParams['ytick.major.width'] = 1

    bins_ = np.arange(-150, 50, 10)
    fig = plt.figure(figsize=(6.85, 8))
    # Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows=4, ncols=3,
                                 figure=fig)

    ###########################
    # Example failrate plots
    ##############################
    spec_failex = gridspec.GridSpecFromSubplotSpec(
        3, 1, spec_all[0, 1], hspace=0.02)
    failex = []

    pr = [0.05, 0.5, 0.95]
    color_pr = [[0.459, 0.631, 0.851],
                [0.3, 0.3, 0.3],
                [0.902, 0.490, 0.322]]
    alpha_ = 1
    alphalow_ = 0.9

    for ind, pr_ in enumerate(pr):
        failex.append(fig.add_subplot(spec_failex[ind, 0]))

        currents = np.where(np.random.rand((50)) < (1 - pr_), 0.0, -10.0)
        currents += (np.random.rand(len(currents)) - 0.5) * 2

        suc_ind = np.where(np.abs(currents) > 5)[0]
        fail_ind = np.where(np.abs(currents) < 5)[0]

        failex[ind].plot(suc_ind, currents[suc_ind], '.',
                         color=color_pr[ind], alpha=alpha_,
                         markeredgewidth=0)
        failex[ind].plot(fail_ind, currents[fail_ind], '.',
                         color=[0.7, 0.7, 0.7], alpha=alpha_,
                         markeredgewidth=0)
        failex[ind].set_ylim([-12, 2])
        failex[ind].set_yticks([-10, 0])
        failex[ind].set_xlim([0, 50])
        failex[ind].set_xticks([])

        text_ = 'Pr = {}'.format(pr_)

        failex[ind].text(0.95,
                         0.5,
                         text_,
                         fontsize=fontsize,
                         fontweight='bold',
                         color=color_pr[ind],
                         alpha=alpha_,
                         transform=failex[ind].transAxes,
                         horizontalalignment='right',
                         verticalalignment='center')

        if ind is 2:
            failex[ind].set_xlabel('Sweeps')
            failex[ind].spines['bottom'].set_visible(True)
            failex[ind].xaxis.set_ticks_position('bottom')
            failex[ind].set_xticks([0, 25, 50])
            sns.despine(ax=failex[ind], offset={'left': 5})
        else:
            failex[ind].axes.get_xaxis().set_visible(False)
            failex[ind].spines['bottom'].set_visible(False)
            sns.despine(ax=failex[ind], bottom=True,
                        offset={'left': 5})

        if ind is 1:
            failex[ind].set_ylabel('Sim. EPSC (pA)')

    ##########################
    # Failure rate dists
    ########################
    fail_dist = fig.add_subplot(spec_all[0, 2])
    fd_inset = inset_axes(fail_dist,
                          width='100%',
                          height='100%',
                          loc=3,
                          bbox_to_anchor=(0.5, 0.75, .4, .25),
                          bbox_transform=fail_dist.transAxes,
                          borderpad=0)
    fd_inset.patch.set_alpha(0)

    # Calculate the failrate dists
    fails = np.empty(len(pr), dtype=np.ndarray)

    for ind, pr_ in enumerate(pr):

        fails[ind] = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis=0) / 50
        fail_dist.hist(fails[ind],
                       bins=np.linspace(0, 1, 100),
                       weights=np.ones_like(fails[ind]) / len(fails[ind]),
                       facecolor=color_pr[ind],
                       alpha=alpha_)

    fail_dist.legend(labels=pr, title='Pr', loc=(0.15, 0.35))
    fail_dist.set_xlabel('failure rate')
    fail_dist.set_xlim([0, 1])
    fail_dist.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    fail_dist.set_ylabel('pdf')
    fail_dist.set_ylim([0, 0.3])
    fail_dist.set_yticks(np.linspace(0, 0.3, 7))

    # Calculate the failrate sd
    inter_ = 0.01
    pr_fine = np.arange(inter_, 1 - inter_, inter_)

    fails_sd = np.empty(len(pr_fine))
    calc_sd = np.empty(len(pr_fine))

    for ind, pr_ in enumerate(pr_fine):
        fails_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis=0) / 50
        fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis=0) / 50

        calc_ = fra(fails_, fails_dp_) * 100
        calc_ = calc_[~np.isnan(calc_)]
        calc_ = calc_[~np.isinf(calc_)]

        fails_sd[ind] = np.std(fails_)
        calc_sd[ind] = np.std(calc_)

    fd_inset.plot(pr_fine, fails_sd, color=[0, 0, 0, 0.5])
    fd_inset.set_xlabel('failure rate')
    fd_inset.set_ylabel('std dev')
    fd_inset.set_ylim([0, 0.08])
    fd_inset.set_xlim([0, 1])
    sns.despine(ax=fail_dist, offset={'left': 5})

    ########################
    # Silent synapse examples
    ########################
    spec_calcex = gridspec.GridSpecFromSubplotSpec(
        3, 1, spec_all[1, 0], hspace=0.02)
    calcex = []

    for ind, pr_ in enumerate(pr):
        calcex.append(fig.add_subplot(spec_calcex[ind, 0]))

        # Simulate failures over one experiment
        fails_hyp_bool = np.random.rand(50) > pr_
        fails_dep_bool = np.random.rand(50) > pr_

        fails_hyp_I = np.where(fails_hyp_bool == True, 0.0, -10.0)
        fails_hyp_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        fails_dep_I = np.where(fails_dep_bool == True, 0.0, 10.0)
        fails_dep_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        # Estimate silent synapse frac
        fails_hyp = np.sum(fails_hyp_bool, axis=0) / 50
        fails_dep = np.sum(fails_dep_bool, axis=0) / 50
        calc_ = (1 - np.log(fails_hyp) / np.log(fails_dep)) * 100

        # Plot the simulated currents
        suc_ind_hyp = np.where(np.abs(fails_hyp_I) > 5)[0]
        fail_ind_hyp = np.where(np.abs(fails_hyp_I) < 5)[0]
        suc_ind_dep = np.where(np.abs(fails_dep_I) > 5)[0]
        fail_ind_dep = np.where(np.abs(fails_dep_I) < 5)[0]

        calcex[ind].plot(suc_ind_hyp,
                         fails_hyp_I[suc_ind_hyp],
                         '.',
                         color=color_pr[ind],
                         alpha=alphalow_,
                         markeredgewidth=0)
        calcex[ind].plot(fail_ind_hyp,
                         fails_hyp_I[fail_ind_hyp],
                         '.',
                         color=[0.7, 0.7, 0.7],
                         alpha=alphalow_,
                         markeredgewidth=0)
        calcex[ind].plot(suc_ind_dep + 50,
                         fails_dep_I[suc_ind_dep],
                         '.',
                         color=color_pr[ind],
                         alpha=alphalow_,
                         markeredgewidth=0)
        calcex[ind].plot(fail_ind_dep + 50,
                         fails_dep_I[fail_ind_dep],
                         '.',
                         color=[0.7, 0.7, 0.7],
                         alpha=alphalow_,
                         markeredgewidth=0)

        calcex[ind].set_ylim([-12, 12])
        calcex[ind].set_yticks([-10, 0, 10])
        calcex[ind].set_xlim([0, 100])
        calcex[ind].spines['bottom'].set_visible(True)
        calcex[ind].xaxis.set_ticks_position('bottom')
        calcex[ind].set_xticks([0, 25, 50, 75, 100])

        text_pr = 'Pr = {}'.format(pr_)
        text_sscalc = 'Est. silent = {0:.1f} %'.format(calc_)

        calcex[ind].text(0.05,
                         0.8,
                         text_pr,
                         fontsize=fontsize,
                         fontweight='bold',
                         color=color_pr[ind],
                         alpha=alphalow_,
                         transform=calcex[ind].transAxes,
                         horizontalalignment='left',
                         verticalalignment='center')
        calcex[ind].text(0.95,
                         0.2,
                         text_sscalc,
                         fontsize=fontsize - 2,
                         fontweight='bold',
                         color='k',
                         transform=calcex[ind].transAxes,
                         horizontalalignment='right',
                         verticalalignment='center')

        if ind is 2:
            calcex[ind].set_xlabel('Sweeps')
            calcex[ind].spines['bottom'].set_visible(True)
            calcex[ind].xaxis.set_ticks_position('bottom')
            calcex[ind].set_xticks([0, 25, 50, 75, 100])
            sns.despine(ax=calcex[ind], offset={'left': 5})
        else:
            calcex[ind].axes.get_xaxis().set_visible(False)
            calcex[ind].spines['bottom'].set_visible(False)
            sns.despine(ax=calcex[ind], bottom=True,
                        offset={'left': 5})

        if ind is 1:
            calcex[ind].set_ylabel('Sim. EPSC (pA)')


    ########################
    # Estimated silent as Pr varies
    ########################
    est_dist = fig.add_subplot(spec_all[1, 1])
    est_cumul = fig.add_subplot(spec_all[1, 2])

    ed_inset = inset_axes(est_dist,
                          width='100%',
                          height='100%',
                          loc=3,
                          bbox_to_anchor=(0.5, 0.75, .4, .25),
                          bbox_transform=est_dist.transAxes,
                          borderpad=0)
    ed_inset.patch.set_alpha(0)

    # Calculate the silent synapse dists
    for ind, pr_ in enumerate(pr):
        fails_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis=0) / 50
        fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_), axis=0) / 50
        calc_ = ((1 - np.log(fails_) / np.log(fails_dp_)) * 100)
        calc_ = calc_[~np.isnan(calc_)]
        calc_ = calc_[~np.isinf(calc_)]

        est_dist.hist(calc_,
                      bins=bins_,
                      weights=np.ones_like(calc_) / len(calc_),
                      color=color_pr[ind],
                      alpha=alpha_,
                      histtype='step')

        est_cumul.hist(calc_,
                       bins=bins_,
                       density=True,
                       histtype='step',
                       cumulative=True,
                       color=color_pr[ind],
                       alpha=alpha_,
                       linewidth=0.8)

    est_dist.legend(labels=pr, title='Pr', loc=(0.15, 0.35))
    est_dist.set_xlabel('est. silent (%)')
    est_dist.set_xlim([-300, 100])
    est_dist.set_xticks([-300, -200, -100, 0, 100])
    est_dist.set_ylabel('pdf')
    est_dist.set_ylim([0, 0.4])
    est_dist.set_yticks(np.linspace(0, 0.4, 5))

    # Calculate the failrate sd
    ed_inset.plot(pr_fine, calc_sd, color=[0, 0, 0, 0.5])
    ed_inset.set_xlabel('Pr')
    ed_inset.set_ylabel('std dev')
    ed_inset.set_ylim([0, 150])
    ed_inset.set_xlim([0, 1])

    est_cumul.legend(labels=pr, title='Pr')
    est_cumul.set_xlabel('est. silent (%)')
    est_cumul.set_xlim([-500, 100])
    est_cumul.set_xticks([-500, -300, -100, 100])
    est_cumul.set_ylabel('cdf')
    est_cumul.set_ylim([0, 1])
    est_cumul.set_yticks(np.linspace(0, 1, 5))

    sns.despine(ax=est_dist, offset={'left': 5})
    sns.despine(ax=est_cumul, offset={'left': 5})

    ########################
    # multi-synapse failure rate
    ########################
    est_dist_multi = fig.add_subplot(spec_all[2, 1])
    est_dist_multi_failx = fig.add_subplot(spec_all[2, 2])

    n_syn = np.arange(1, 7)
    calc_syn = np.empty_like(n_syn, dtype=np.ndarray)
    color_syn = np.empty_like(n_syn, dtype=np.ndarray)
    failrate_syn = np.empty_like(n_syn, dtype=np.ndarray)

    for ind, n in enumerate(n_syn):
        calc_syn[ind] = np.empty_like(pr_fine)
        failrate_syn[ind] = np.empty_like(pr_fine)

        # Pick a color
        color_syn[ind] = cmap_n(ind / (len(n_syn) - 1))

        for ind_pr, pr_ in enumerate(pr_fine):
            failrate_syn[ind][ind_pr] = (1 - pr_)**n

            fails_ = np.sum(np.random.rand(50, 10000) <
                            (1 - pr_)**n, axis=0) / 50
            fails_dp_ = np.sum(np.random.rand(50, 10000) < (1 - pr_)**n,
                               axis=0) / 50
            calc_ = ((1 - np.log(fails_) / np.log(fails_dp_)) * 100)
            calc_ = calc_[~np.isnan(calc_)]
            calc_ = calc_[~np.isinf(calc_)]

            calc_syn[ind][ind_pr] = np.std(calc_)

        # Postprocessing on the data
        find_range = np.where(
            np.logical_or(failrate_syn[ind] < 0.05, failrate_syn[ind] > 0.95))
        calc_syn[ind][find_range] = np.nan

        est_dist_multi.plot(pr_fine,
                            calc_syn[ind],
                            color=color_syn[ind],
                            alpha=alpha_,
                            linewidth=1)
        est_dist_multi_failx.plot(failrate_syn[ind],
                                  calc_syn[ind],
                                  color=color_syn[ind],
                                  alpha=alpha_,
                                  linewidth=1)

    est_dist_multi.legend(labels=n_syn, title='Num. synapses')
    est_dist_multi.set_xlabel('release probability')
    est_dist_multi.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    est_dist_multi.set_ylabel('stdev(est. silent) (%)')
    est_dist_multi.set_ylim([0, 150])
    est_dist_multi.set_yticks(np.arange(0, 151, 25))
    sns.despine(ax=est_dist_multi, offset={'left': 5})

    est_dist_multi_failx.legend(labels=n_syn, title='Num. synapses')
    est_dist_multi_failx.set_xlabel('failure rate')
    est_dist_multi_failx.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    est_dist_multi_failx.set_ylabel('stdev(est. silent) (%)')
    est_dist_multi_failx.set_ylim([0, 150])
    est_dist_multi_failx.set_yticks(np.arange(0, 151, 25))
    sns.despine(ax=est_dist_multi_failx, offset={'left': 5})

    ###########################
    # Delta failure rate plots
    ##############################
    # spec_failex = gridspec.GridSpecFromSubplotSpec(
    #     3, 1, spec_all[3, 0], hspace=0.02)
    spec_failex = spec_all[3, 0].subgridspec(3, 1)
    failex = []

    fails_hyp = [0.45, 0.05, 0.85]
    fails_dep = [0.55, 0.15, 0.95]

    color_fails = np.empty(len(fails_hyp), dtype=np.ndarray)

    for ind, fails_hyp_ in enumerate(fails_hyp):

        failex.append(fig.add_subplot(spec_failex[ind, 0]))

        # First, calculate the color to use
        # ratio_ = ind / (len(fails_hyp) - 1)
        # blue_ = 1 / (1 + np.exp(-5 * (ratio_ - 0.5)))
        # color_ = [0, 0.25 + blue_ / 3, 0.7 - (blue_ / 2)]
        # color_fails[ind] = color_
        color_fails[ind] = cmap_n((ind+1)/len(fails_hyp))

        # Simulate failures over one experiment
        fails_hyp_bool = np.zeros(50, dtype=np.bool)
        fails_hyp_bool[0:int(fails_hyp[ind] * 50)] = True
        np.random.shuffle(fails_hyp_bool)

        fails_dep_bool = np.zeros(50, dtype=np.bool)
        fails_dep_bool[0:int(fails_dep[ind] * 50)] = True
        np.random.shuffle(fails_dep_bool)

        fails_hyp_I = np.where(fails_hyp_bool == True, 0.0, -10.0)
        fails_hyp_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        fails_dep_I = np.where(fails_dep_bool == True, 0.0, 10.0)
        fails_dep_I += (np.random.rand(len(fails_hyp_I)) - 0.5) * 2

        # Estimate silent synapse frac
        calc_ = (1 - np.log(fails_hyp[ind]) / np.log(fails_dep[ind])) * 100

        # Plot the simulated currents
        suc_ind_hyp = np.where(np.abs(fails_hyp_I) > 5)[0]
        fail_ind_hyp = np.where(np.abs(fails_hyp_I) < 5)[0]
        suc_ind_dep = np.where(np.abs(fails_dep_I) > 5)[0]
        fail_ind_dep = np.where(np.abs(fails_dep_I) < 5)[0]

        failex[ind].plot(suc_ind_hyp,
                         fails_hyp_I[suc_ind_hyp],
                         '.',
                         color=color_fails[ind],
                         alpha=alpha_,
                         markeredgewidth=0)
        failex[ind].plot(fail_ind_hyp,
                         fails_hyp_I[fail_ind_hyp],
                         '.',
                         color=[0.7, 0.7, 0.7],
                         alpha=alpha_,
                         markeredgewidth=0)
        failex[ind].plot(suc_ind_dep + 50,
                         fails_dep_I[suc_ind_dep],
                         '.',
                         color=color_fails[ind],
                         alpha=alpha_,
                         markeredgewidth=0)
        failex[ind].plot(fail_ind_dep + 50,
                         fails_dep_I[fail_ind_dep],
                         '.',
                         color=[0.7, 0.7, 0.7],
                         alpha=alpha_,
                         markeredgewidth=0)

        failex[ind].set_ylim([-16, 16])
        failex[ind].set_yticks([-10, 0, 10])
        failex[ind].set_xlim([-15, 100])
        failex[ind].spines['bottom'].set_visible(True)
        failex[ind].xaxis.set_ticks_position('bottom')
        failex[ind].set_xticks([0, 25, 50, 75, 100])


        text_hp = 'hyp. fails: {0:.2f}'.format(fails_hyp[ind])
        text_dp = 'dep. fails: {0:.2f}'.format(fails_dep[ind])

        failex[ind].text(0.05,
                         0.8,
                         text_hp,
                         fontsize=fontsize,
                         fontweight='bold',
                         color=color_fails[ind],
                         alpha=alpha_ - 0.2,
                         transform=failex[ind].transAxes,
                         horizontalalignment='left',
                         verticalalignment='center')
        failex[ind].text(0.95,
                         0.2,
                         text_dp,
                         fontsize=fontsize,
                         fontweight='bold',
                         color=color_fails[ind],
                         alpha=alpha_ - 0.2,
                         transform=failex[ind].transAxes,
                         horizontalalignment='right',
                         verticalalignment='center')

        if ind is 2:
            failex[ind].set_xlabel('Sweeps')
            failex[ind].spines['bottom'].set_visible(True)
            failex[ind].xaxis.set_ticks_position('bottom')
            failex[ind].set_xticks([0, 25, 50, 75, 100])
            sns.despine(ax=failex[ind], offset={'left': 5})
        else:
            failex[ind].spines['bottom'].set_visible(False)
            failex[ind].axes.get_xaxis().set_visible(False)
            sns.despine(ax=failex[ind], bottom=True,
                        offset={'left': 5})

        if ind is 1:
            failex[ind].set_ylabel('Sim. EPSC (pA)')

    # Natural logarithm
    # ##################
    spl_log = fig.add_subplot(spec_all[3, 1])

    fails_fine = np.linspace(0, 1, num=1000)
    log_fails_fine = np.log(fails_fine)
    spl_log.plot(fails_fine, log_fails_fine, color='k')
    spl_log.set_xlabel('failure rate')
    spl_log.set_ylabel('log(failure rate)')
    spl_log.set_ylim([-4, 1])
    spl_log.set_xlim([0, 1])

    for ind, fails_hyp_ in enumerate(fails_hyp):
        spl_log.annotate("",
                         xy=(fails_hyp[ind], np.log(fails_hyp[ind])),
                         xycoords='data',
                         xytext=(fails_hyp[ind], -4),
                         textcoords='data',
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="arc3",
                                         color=color_fails[ind],
                                         alpha=0.9,
                                         linewidth=1,
                                         shrinkA=1,
                                         shrinkB=0.5))

        spl_log.annotate("",
                         xy=(fails_dep[ind], np.log(fails_dep[ind])),
                         xycoords='data',
                         xytext=(fails_dep[ind], -4),
                         textcoords='data',
                         arrowprops=dict(arrowstyle="->",
                                         connectionstyle="arc3",
                                         color=color_fails[ind],
                                         alpha=0.9,
                                         linewidth=1,
                                         shrinkA=1,
                                         shrinkB=0.5))

        spl_log.annotate("",
                         xy=(fails_dep[ind], -3.4),
                         xycoords='data',
                         xytext=(fails_hyp[ind], -3.4),
                         textcoords='data',
                         arrowprops=dict(arrowstyle="<->",
                                         connectionstyle="arc3",
                                         color='k',
                                         alpha=0.9,
                                         linewidth=1,
                                         shrinkA=0.5,
                                         shrinkB=0.2))

        xcoord_text_ = (fails_hyp[ind] + fails_dep[ind]) / 2
        ycoord_text_ = -3.85

        spl_log.text(xcoord_text_,
                     ycoord_text_,
                     '$\Delta$ 0.1',
                     fontsize=fontsize - 1,
                     color='k',
                     alpha=alpha_ - 0.2,
                     horizontalalignment='center',
                     verticalalignment='center')

        # Fill in between the arrows
        xrange_fill = np.linspace(fails_hyp[ind], fails_dep[ind], num=100)
        y1_fill = np.log(xrange_fill)
        y2_fill = np.ones_like(xrange_fill) * (np.log(fails_hyp[ind]) - 0.04)

        spl_log.fill_between(xrange_fill,
                             y1_fill,
                             y2_fill,
                             facecolor=color_fails[ind],
                             alpha=0.3)

        est_silent = (1 -
                      np.log(fails_hyp[ind]) / np.log(fails_dep[ind])) * 100
        text_calc = '{0:.1f} % \n est. silent'.format(est_silent)

        # Make text reporting failure rates
        xcoord_text_ = (fails_hyp[ind] + fails_dep[ind]) / 2
        ycoord_text_ = np.log(fails_dep[ind]) + 0.6

        spl_log.text(xcoord_text_,
                     ycoord_text_,
                     text_calc,
                     fontsize=fontsize,
                     fontweight='bold',
                     color=color_fails[ind],
                     alpha=1,
                     horizontalalignment='center',
                     verticalalignment='center')

        sns.despine(ax=spl_log, offset={'left': 5})

    # spec_all.tight_layout(fig)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    fig.savefig(path_f)
    plt.show()

    return

# fig2:
# For gamma:
# gamma_shape = 3
# gamma_rate = 5.8
# gamma_scale = 1 / gamma_rate


def plot_fig2(trueval_lim=0.1,
              frac_reduction=0.1,
              method_='iterative',
              pr_dist_sil=PrDist(sp_stats.uniform),
              pr_dist_nonsil=PrDist(sp_stats.uniform),
              cmap=cm_,
              fontsize=8,
              plot_sims=40,
              despine_offset={'left': 3},
              ylab_pad_tight=-3,
              xlim_n_active=8,
              xlim_n_silent=40,
              figname='Fig2.pdf'):
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

    # -------------------
    # 1: Simulation of synapse groups
    # -------------------
    n_start = 100

    # Run simulation: 0.5 silent
    nonsilent_syn_group_half, silent_syn_group_half, \
        pr_nonsilent_syn_group_half, pr_silent_syn_group_half \
        = draw_subsample(method=method_,
                         pr_dist_sil=pr_dist_sil,
                         pr_dist_nonsil=pr_dist_nonsil,
                         n_simulations=10000, n_start=n_start,
                         silent_fraction=0.5, failrate_low=0.2,
                         failrate_high=0.8,
                         unitary_reduction=False,
                         frac_reduction=frac_reduction)

    # Run simulation for the whole range
    silent_truefrac_coarse = np.arange(0.1, 0.95, 0.2)

    ratio_estimate_silent_array = np.empty(len(silent_truefrac_coarse),
                                           dtype=np.ndarray)

    ratio_estimate_silent_mean = np.empty(len(silent_truefrac_coarse))
    ratio_estimate_silent_std = np.empty(len(silent_truefrac_coarse))

    nonsilent_syn_group = np.empty(len(silent_truefrac_coarse),
                                   dtype=np.ndarray)
    silent_syn_group = np.empty_like(nonsilent_syn_group)
    pr_nonsilent_syn_group = np.empty_like(nonsilent_syn_group)
    pr_silent_syn_group = np.empty_like(nonsilent_syn_group)

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        print('\nSilent fraction: ', str(silent))

        nonsilent_syn_group[ind_silent], silent_syn_group[ind_silent], \
            pr_nonsilent_syn_group[ind_silent], pr_silent_syn_group[ind_silent] \
            = draw_subsample(method=method_,
                             pr_dist_sil=pr_dist_sil,
                             pr_dist_nonsil=pr_dist_nonsil,
                             n_simulations=10000, n_start=n_start,
                             silent_fraction=silent, failrate_low=0.2,
                             failrate_high=0.8,
                             unitary_reduction=False,
                             frac_reduction=frac_reduction)

        ratio_estimate_silent_array[ind_silent] = silent_syn_group[ind_silent]\
            / (nonsilent_syn_group[ind_silent] + silent_syn_group[ind_silent])

        ratio_estimate_silent_mean[ind_silent] = np.mean(
            ratio_estimate_silent_array[ind_silent])
        ratio_estimate_silent_std[ind_silent] = np.std(
            ratio_estimate_silent_array[ind_silent])

    # -------------------
    # 2: Simulation of FRA values returned
    # -------------------
    silent_truefrac_fine = np.arange(0, 0.55, 0.01)

    fra_calc = np.empty(len(silent_truefrac_fine), dtype=np.ndarray)
    fra_calc_z = np.empty(len(silent_truefrac_fine),
                          dtype=np.ndarray)  # zeroing

    # Store fraction of sims reaching true val, and the error of the mean sims
    # in small samples of n = 10, 20, 30.
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

        # Make calculation of FRA values
        fra_calc[ind] = gen_fra_dist(method=method_,
                                     pr_dist_sil=pr_dist_sil,
                                     pr_dist_nonsil=pr_dist_nonsil,
                                     n_simulations=10000,
                                     silent_fraction=silent,
                                     zeroing=False,
                                     unitary_reduction=False,
                                     frac_reduction=frac_reduction)

        fra_calc_z[ind] = gen_fra_dist(method=method_,
                                       pr_dist_sil=pr_dist_sil,
                                       pr_dist_nonsil=pr_dist_nonsil,
                                       n_simulations=10000,
                                       silent_fraction=silent,
                                       zeroing=True,
                                       unitary_reduction=False,
                                       frac_reduction=frac_reduction)

        # Store fraction of 'true' estimates
        error_ = np.abs(fra_calc[ind] - silent_truefrac_fine[ind])
        frac_true[ind] = len(np.where(error_ < trueval_lim)[0]) / len(error_)

        error_z_ = np.abs(fra_calc_z[ind] - silent_truefrac_fine[ind])
        frac_true_z[ind] = len(
            np.where(error_z_ < trueval_lim)[0]) / len(error_)

        # Draw subsets of 10, 20, 30 samples from FRA and store means
        draws_n10_ = np.empty(1000)
        draws_n20_ = np.empty(1000)
        draws_n30_ = np.empty(1000)

        draws_n10_z_ = np.empty(1000)
        draws_n20_z_ = np.empty(1000)
        draws_n30_z_ = np.empty(1000)

        for draw in range(1000):
            inds_n10_ = np.random.randint(0, 10000, size=10)
            inds_n20_ = np.random.randint(0, 10000, size=20)
            inds_n30_ = np.random.randint(0, 10000, size=30)

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

    # -----------------------------PLOTTING------------------------
    try:
        plt.style.use('publication_ml')
    except FileNotFoundError:
        pass
    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(5)
    fig.set_figwidth(3.43)

    # Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows=6, ncols=6, figure=fig,
                                 height_ratios=[1, 1, 1, 1, 0.5, 0.5])

    # Define colors to be used
    color_nonsilent = np.array([0.1, 0.55, 0.85])
    color_silent = np.array([0.76, 0.26, 0.22])

    color_pr = np.array([0.25, 0.55, 0.18])

    # Subplot 2: Number of active/silent synapses
    n_plot_nonsil = fig.add_subplot(spec_all[0, 2:4])
    n_plot_sil = fig.add_subplot(spec_all[0, 4:6], sharey=n_plot_nonsil)
    # ratio_accuracy = fig.add_subplot(spec_bottomleft[:, 2])

    count_nonsil = np.empty(len(silent_truefrac_coarse), dtype=np.ndarray)
    count_sil = np.empty(len(silent_truefrac_coarse), dtype=np.ndarray)

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        count_nonsil[ind_silent] = np.bincount(nonsilent_syn_group[ind_silent])
        count_nonsil[ind_silent] = count_nonsil[ind_silent] \
            / np.sum(count_nonsil[ind_silent])

        count_sil[ind_silent] = np.bincount(silent_syn_group[ind_silent])
        count_sil[ind_silent] = count_sil[ind_silent] / np.sum(
            count_sil[ind_silent])

        n_plot_nonsil.plot(np.arange(1, len(count_nonsil[ind_silent])),
                           count_nonsil[ind_silent][1:],
                           color=color_nonsilent * silent)
        n_plot_sil.plot(np.arange(0, len(count_sil[ind_silent])),
                        count_sil[ind_silent],
                        color=color_silent * silent)

    n_plot_nonsil.set_xlabel('active synapses')
    n_plot_nonsil.set_ylabel('pdf')
    n_plot_nonsil.set_xticks(np.arange(0, xlim_n_active+1, 2))
    n_plot_nonsil.set_xlim([0, xlim_n_active])
    n_plot_nonsil.set_ylim([0, n_plot_nonsil.get_ylim()[1]])
    sns.despine(ax=n_plot_nonsil, offset=despine_offset)

    leg_ns = n_plot_nonsil.legend(labels=(silent_truefrac_coarse * 100)
                                   .astype(np.int),
                                  title='silent (%)',
                                  ncol=2,
                                  loc=1,
                                  handlelength=0.5,
                                  fontsize='x-small')

    n_plot_sil.set_xlabel('silent synapses')
    n_plot_sil.set_xticks(np.arange(0, xlim_n_silent+1, 10))
    n_plot_sil.set_xlim([0, xlim_n_silent])
    n_plot_sil.set_ylim([0, n_plot_sil.get_ylim()[1]])
    sns.despine(ax=n_plot_sil, offset=despine_offset)

    leg_s = n_plot_sil.legend(
        labels=(silent_truefrac_coarse * 100).astype(
            np.int),
        title='silent (%)',
        ncol=2,
        loc=1,
        handlelength=0.5,
        fontsize='x-small')

    # Subplot 4/5: Nonsil/sil Release probability dist.:
    release_prob_nonsil = fig.add_subplot(spec_all[1, 2:4])
    subset_error = fig.add_subplot(spec_all[1, 4:6])

    max_n_nonsilent = len(np.bincount(nonsilent_syn_group_half))
    max_n_silent = len(np.bincount(silent_syn_group_half))

    pr_nonsil = np.empty(max_n_nonsilent, dtype=np.ndarray)
    pr_nonsil_mean = np.empty(max_n_nonsilent)
    n_pr_toplot = 100

    # Plot mean release probability for each number of nonsilent/silent syns
    for syn_number in range(1, max_n_nonsilent):
        inds_with_syn_number = np.where(
            pr_nonsilent_syn_group_half.count(axis=1) == syn_number)
        pr_nonsil[syn_number] = np.ma.mean(
            pr_nonsilent_syn_group_half[inds_with_syn_number], axis=1)
        pr_nonsil_mean[syn_number] = np.mean(pr_nonsil[syn_number])

        if pr_nonsil[syn_number] is not None:
            release_prob_nonsil.plot(
                np.ones(len(pr_nonsil[syn_number][0:n_pr_toplot])) *
                syn_number,
                pr_nonsil[syn_number][0:n_pr_toplot],
                '.',
                color=color_pr,
                alpha=0.1,
                markersize=1)
            release_prob_nonsil.plot(
                syn_number, pr_nonsil_mean[syn_number], '.',
                color=color_pr, markersize=5)

    # Plot the subset composition vs superset composition for each position
    subset_fracsil = np.empty(len(silent_truefrac_coarse), dtype=np.ndarray)
    subset_fracsil_mean = np.empty(len(silent_truefrac_coarse))
    subset_fracsil_sd = np.empty(len(silent_truefrac_coarse))

    for ind_silent, silent in enumerate(silent_truefrac_coarse):
        subset_fracsil[ind_silent] = silent_syn_group[ind_silent] \
            / (nonsilent_syn_group[ind_silent] + silent_syn_group[ind_silent])

        subset_fracsil_mean[ind_silent] = np.mean(subset_fracsil[ind_silent])
        subset_fracsil_sd[ind_silent] = np.std(subset_fracsil[ind_silent])

    subset_error.plot(silent_truefrac_coarse,
                      subset_fracsil_mean,
                      color=color_silent,
                      alpha=0.9)
    subset_error.fill_between(silent_truefrac_coarse,
                              subset_fracsil_mean - subset_fracsil_sd,
                              subset_fracsil_mean + subset_fracsil_sd,
                              facecolor=color_silent,
                              alpha=0.2)

    subset_error.plot(np.linspace(0, 1, 10),
                      np.linspace(0, 1, 10),
                      '--k',
                      linewidth=0.5)

    # Adjust plot features
    release_prob_nonsil.set_xlim([0, xlim_n_active+0.5])
    release_prob_nonsil.set_ylim([0, 1])
    release_prob_nonsil.set_xticks(np.arange(0, xlim_n_active+1, 2))
    release_prob_nonsil.set_yticks([0, 0.5, 1])
    release_prob_nonsil.set_xlabel('active synapses')
    release_prob_nonsil.set_ylabel('mean Pr')
    sns.despine(ax=release_prob_nonsil, offset=despine_offset)

    subset_error.set_xlim([0, 1])
    subset_error.set_ylim([0, 1])
    subset_error.set_xticks([0, 0.5, 1])
    subset_error.set_yticks([0, 0.5, 1])
    subset_error.set_xlabel('frac. silent')
    subset_error.set_ylabel('sampled frac. sil.')
    sns.despine(ax=subset_error, offset=despine_offset)

    # -------------------
    # 1: Plot example synapse groups and example traces
    # -------------------
    sim_ex_30 = fig.add_subplot(spec_all[2:3, 0:3])
    # Subplot 1: synapsegroups
    img_groups_30 = np.ones((100,
                             np.max(nonsilent_syn_group[1][0:plot_sims] +
                                    silent_syn_group[1][0:plot_sims]), 3))

    for sim in range(plot_sims):
        img_groups_30[sim, 0:nonsilent_syn_group[1][sim], :] = np.tile(
            [0, 0, 0], (1, nonsilent_syn_group[1][sim], 1))
        img_groups_30[sim, nonsilent_syn_group[1][sim]:
                      nonsilent_syn_group[1][sim] +
                      silent_syn_group[1][sim], :] = np.tile(
                          [0.8, 0.1, 0.2], (1, silent_syn_group[1][sim], 1))
    # Plot 30% silent example
    img30 = sim_ex_30.imshow(img_groups_30,
                             aspect='auto',
                             alpha=0.9)

    sim_ex_30.plot()
    sim_ex_30.set_ylabel('simulations')
    sim_ex_30.set_ylim([plot_sims - 0.5, -0.5])
    sim_ex_30.set_yticks([])
    sim_ex_30.set_xticks([0 - 0.5, 5 - 0.5, 10 - 0.5, 15 - 0.5])
    sim_ex_30.set_xticklabels([0, 5, 10, 15])
    sim_ex_30.text(1,
                   0.55,
                   'active',
                   verticalalignment='top',
                   horizontalalignment='right',
                   color=[0, 0, 0],
                   transform=sim_ex_30.transAxes,
                   fontsize=fontsize - 1,
                   fontweight='bold')
    sim_ex_30.text(1,
                   0.45,
                   'silent',
                   verticalalignment='top',
                   horizontalalignment='right',
                   color=cmap(0.99),
                   transform=sim_ex_30.transAxes,
                   fontsize=fontsize - 1,
                   fontweight='bold')

    # -------------------
    # 2: Example traces for various silent fracs
    # #-------------------
    silent_extraces = [0.3]
    failex = []

    for ind, silent_ in enumerate(silent_extraces):
        failex.append(fig.add_subplot(spec_all[3:4, 0:3]))

        nonsilent_syn_group_, silent_syn_group_, \
            pr_nonsilent_syn_group_, pr_silent_syn_group_ \
            = draw_subsample(pr_dist_sil=pr_dist_sil,
                             pr_dist_nonsil=pr_dist_nonsil,
                             n_simulations=1, n_start=n_start,
                             silent_fraction=silent_, failrate_low=0.2,
                             failrate_high=0.8,
                             unitary_reduction=False,
                             frac_reduction=frac_reduction)

        pr_sweeps_hyp_ = np.tile(pr_nonsilent_syn_group_.compressed(),
                                 50).reshape(50, nonsilent_syn_group_[0])
        pr_sweeps_dep_ = np.tile(
            np.append(pr_nonsilent_syn_group_.compressed(),
                      pr_silent_syn_group_.compressed()),
            50).reshape(50, nonsilent_syn_group_[0] + silent_syn_group_[0])

        currents_hyp = np.where(
            np.random.rand(50, nonsilent_syn_group_[0]) < (1 - pr_sweeps_hyp_),
            0.0, -10.0)
        currents_hyp += (np.random.rand(currents_hyp.shape[0],
                                        currents_hyp.shape[1]) - 0.5) * 2
        currents_hyp_sum = np.sum(currents_hyp, axis=1)

        currents_dep = np.where(
            np.random.rand(50, nonsilent_syn_group_[0] + silent_syn_group_[0])
            < (1 - pr_sweeps_dep_), 0.0, 10.0)
        currents_dep += (np.random.rand(currents_dep.shape[0],
                                        currents_dep.shape[1]) - 0.5) * 2
        currents_dep_sum = np.sum(currents_dep, axis=1)

        suc_ind_hyp = np.where(np.abs(currents_hyp_sum) > 5)[0]
        fail_ind_hyp = np.where(np.abs(currents_hyp_sum) < 5)[0]

        suc_ind_dep = np.where(np.abs(currents_dep_sum) > 5)[0]
        fail_ind_dep = np.where(np.abs(currents_dep_sum) < 5)[0]

        est_silent = (1 - np.log(len(fail_ind_hyp) / 50)
                      / np.log(len(fail_ind_dep) / 50)) * 100

        failex[ind].plot(suc_ind_hyp,
                         currents_hyp_sum[suc_ind_hyp],
                         '.',
                         color=cmap(0),
                         markeredgewidth=0)
        failex[ind].plot(fail_ind_hyp,
                         currents_hyp_sum[fail_ind_hyp],
                         '.',
                         color=cmap(0.5),
                         markeredgewidth=0)
        failex[ind].plot(suc_ind_dep + 50,
                         currents_dep_sum[suc_ind_dep],
                         '.',
                         color=cmap(0.99),
                         markeredgewidth=0)
        failex[ind].plot(fail_ind_dep + 50,
                         currents_dep_sum[fail_ind_dep],
                         '.',
                         color=cmap(0.5),
                         markeredgewidth=0)
        failex[ind].plot([0, 100], [0, 0], ':k', linewidth=1)

        failex[ind].set_xlim([0, 100])
        failex[ind].set_xticks([0, 25, 50, 75, 100])
        ymax_ = failex[ind].get_ylim()[1]
        failex[ind].set_ylim([-ymax_, ymax_])
        failex[ind].spines['bottom'].set_visible(True)
        failex[ind].xaxis.set_ticks_position('bottom')
        failex[ind].set_ylabel('sim. current (pA)')
        failex[ind].set_xlabel('sweeps')
        sns.despine(ax=failex[ind], offset=despine_offset)

        text_truesil = '{}% silent'.format(int(silent_ * 100))
        text_estsil = 'Est. silent\n = {0:.1f}%'.format(est_silent)

        failex[ind].text(0.05,
                         0.95,
                         text_truesil,
                         fontweight='bold',
                         color=color_nonsilent * 0.4,
                         alpha=0.9,
                         transform=failex[ind].transAxes,
                         horizontalalignment='left',
                         verticalalignment='top',
                         bbox=dict(facecolor='none',
                                   edgecolor=np.append(color_nonsilent * 0.5,
                                                       0.8),
                                   pad=2.0))

        failex[ind].text(1,
                         0.1,
                         text_estsil,
                         fontweight='bold',
                         color=[0, 0, 0],
                         alpha=1,
                         transform=failex[ind].transAxes,
                         horizontalalignment='right',
                         verticalalignment='bottom')

    # -------------------
    # 3: Plots of FRA values returned
    # -------------------

    #########
    # first, plot estimate distributions
    ######
    histtype_ = 'step'
    desired_silent = 0.3
    ind_desired_silent = np.argmin(
        np.abs(silent_truefrac_fine - desired_silent))
    bins_ = np.linspace(0, 100, 40)

    ax_frahist_z = fig.add_subplot(spec_all[2:4, 3:6])

    # Histogram
    ax_frahist_z.hist(fra_calc_z[0] * 100,
                      bins=bins_,
                      density=True,
                      color=[0, 0, 0, 1],
                      histtype=histtype_)
    ax_frahist_z.hist(fra_calc_z[ind_desired_silent] * 100,
                      bins=bins_,
                      density=True,
                      color=cmap(0.99),
                      histtype=histtype_)
    ax_frahist_z.legend(labels=['0', str(desired_silent * 100)],
                        title='silent (%)')

    ax_frahist_z.set_xlim([0, 100])
    ax_frahist_z.set_xticks([0, 25, 50, 75, 100])
    ax_frahist_z.set_xlabel('estimated silent (%)')
    ax_frahist_z.set_ylabel('pdf')
    sns.despine(ax=ax_frahist_z, offset=despine_offset)

    # -------------------
    # 4: Bias and variance of estimator
    # -------------------
    ax_bias = fig.add_subplot(spec_all[4:6, 0:3])

    bias = np.empty(len(silent_truefrac_fine))
    bias_no_z = np.empty(len(silent_truefrac_fine))
    for ind, frac in enumerate(silent_truefrac_fine):
        bias[ind] = (np.mean(fra_calc_z[ind]) - silent_truefrac_fine[ind])
        bias_no_z[ind] = (np.mean(fra_calc[ind]) - silent_truefrac_fine[ind])
    ax_bias.plot(silent_truefrac_fine * 100,
                 bias * 100,
                 color=[0, 0, 0])
    ax_bias.plot(silent_truefrac_fine * 100,
                 bias_no_z * 100,
                 color=[0.5, 0.5, 0.5])
    ax_bias.legend(labels=['FRA (zeroed)', 'FRA (raw)'], title='estimator')
    ax_bias.set_xlabel('true silent (%)')
    ax_bias.set_ylabel('estimator bias (%)')
    ax_bias.set_xticks([0, 10, 20, 30, 40, 50])
    ax_bias.set_xlim([0, 50])
    ax_bias.set_yticks([0, 10, 20, 30])
    sns.despine(ax=ax_bias, offset=despine_offset)

    ax_var = fig.add_subplot(spec_all[4:6, 3:6])

    stdev = np.empty(len(silent_truefrac_fine))
    stdev_no_z = np.empty(len(silent_truefrac_fine))
    for ind, frac in enumerate(silent_truefrac_fine):
        stdev[ind] = np.std(fra_calc_z[ind] * 100)
        stdev_no_z[ind] = np.std(fra_calc[ind] * 100)
    ax_var.plot(silent_truefrac_fine * 100,
                stdev,
                color=[0, 0, 0])
    ax_var.plot(silent_truefrac_fine * 100,
                stdev_no_z,
                color=[0.5, 0.5, 0.5])
    ax_var.legend(labels=['FRA (zeroed)', 'FRA (raw)'], title='estimator')
    ax_var.set_xlabel('true silent (%)')
    ax_var.set_ylabel('estimator std (%)')
    ax_var.set_ylim([0, ax_var.get_ylim()[1]])
    ax_var.set_xticks([0, 10, 20, 30, 40, 50])
    ax_var.set_xlim([0, 50])
    sns.despine(ax=ax_var, offset=despine_offset)

    # Set tight layouts for all
    # spec_all.tight_layout(fig)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    plt.savefig(path_f)

    return


def plot_figS2():
    """Plots figure S2 (experimental simulator, with physiologically
    realistic release probability distribution of gamma (shape: 3,
    scale: 1/5.8)
    """
    pr_dist_gamma = PrDist(sp_stats.gamma,
                           args={'a': 3,
                                 'scale': 1/5.8})

    plot_fig2(figname='FigS2.pdf',
              pr_dist_sil=pr_dist_gamma,
              pr_dist_nonsil=pr_dist_gamma)

    return


def plot_figS3():
    """Plots figure S3 (experimental simulator, with extreme
    release probability distribution of gamma (shape: 1,
    scale: 1/5.8)
    """
    pr_dist_gamma = PrDist(sp_stats.gamma,
                           args={'a': 1,
                                 'scale': 1/5.8})

    plot_fig2(figname='FigS3.pdf',
              pr_dist_sil=pr_dist_gamma,
              pr_dist_nonsil=pr_dist_gamma,
              xlim_n_active=16,
              xlim_n_silent=100)

    return


def plot_fig4(n_true_silents_power=26,
              n_true_silents_mle=500,
              sample_draws=5000,
              fontsize=8,
              despine_offset={'left': 5},
              figname='Fig4.pdf'):
    '''
    Plot the FRA_MLE estimator and power analysis figure.
    '''

    # 1. Simulations for power analysis
    ##########################################################

    # Generate fra and binary distributions
    # ----------------------------------------------
    silent_truefrac = np.linspace(0, 0.5, num=n_true_silents_power)
    fra_calc = np.empty(len(silent_truefrac), dtype=np.ndarray)
    binary_calc = np.empty_like(fra_calc)

    # Set up multiple conditions
    condlist = ['base', 'binary',
                'h0_fra', 'h0_binary',
                'h0_llr_framle', 'h0_llr_binary']
    condlist_discrim = condlist[0:2]
    condlist_h0 = condlist[2:]

    # Set parameters for each condition
    conds_beta = {cond: 0.2 for cond in condlist}
    conds_ctrl_n = {cond: False for cond in condlist}

    mins_templ = np.empty((len(silent_truefrac), len(silent_truefrac)))
    mins_1d_templ = np.empty((len(silent_truefrac)))

    minsamples = {cond: mins_templ.copy() for cond in condlist_discrim}
    for cond in condlist_h0:  # cases for hypothesis testing (h0 compar)
        minsamples[cond] = mins_1d_templ.copy()

    print('Generating FRA calcs...')
    # Generate FRA calcs and calc simple minsamples versus baseline
    for ind, silent in enumerate(silent_truefrac):
        print(f'\tsilent: {silent*100}%', end='\r')
        # Generate an FRA dist
        fra_calc[ind] = gen_fra_dist(n_simulations=10000,
                                     silent_fraction=silent,
                                     zeroing=False,
                                     unitary_reduction=False,
                                     frac_reduction=0.2)

        binary_calc[ind] = gen_fra_dist(n_simulations=10000,
                                        silent_fraction=silent,
                                        zeroing=False,
                                        unitary_reduction=False,
                                        frac_reduction=0.2,
                                        binary_vals=True)

    # Set up guesses for each condition to start at
    guess = {cond: int(2048) for cond in condlist}

    # Minsamples[discrim.]: Standard FRA with numerical sims
    # ----------------------------------
    for ind1_sil, sil1 in enumerate([0]):
        for ind2_sil, sil2 in enumerate(silent_truefrac):

            # Only run power analysis for upper right triangular matrix
            if ind2_sil > ind1_sil:
                print('\nsil_1 ' + str(sil1) + '; sil_2 ' + str(sil2))

                # Iterate through all conditions with discrimin.
                for cond in condlist_discrim:
                    print('\n\tcond: ' + str(cond))

                    # Update guesses
                    if ind2_sil == ind1_sil + 1 or guess[cond] == 2048:
                        guess[cond] = np.int(2048)
                    else:
                        # Ctrl group
                        exp_2 = np.ceil(
                            np.log(minsamples[cond][ind1_sil, ind2_sil - 1]) /
                            np.log(2))
                        guess[cond] = np.int(2**(exp_2))

                    # Calculate minsamples based on cond
                    if cond == 'binary':
                        minsamples[cond][ind1_sil, ind2_sil] = power_analysis(
                            binary_calc[ind1_sil],
                            binary_calc[ind2_sil],
                            init_guess=guess[cond],
                            beta=conds_beta[cond],
                            ctrl_n=conds_ctrl_n[cond],
                            sample_draws=sample_draws,
                            stat_test='chisquare')
                    else:
                        minsamples[cond][ind1_sil, ind2_sil] = power_analysis(
                            fra_calc[ind1_sil],
                            fra_calc[ind2_sil],
                            init_guess=guess[cond],
                            beta=conds_beta[cond],
                            ctrl_n=conds_ctrl_n[cond],
                            sample_draws=sample_draws)

                    guess[cond] = minsamples[cond][ind1_sil, ind2_sil]

                    # Set mirror image on other side to same value
                    minsamples[cond][ind2_sil, ind1_sil] = \
                        minsamples[cond][ind1_sil, ind2_sil]

                # Set same-same comparisons to arbitrarily high val
            elif ind2_sil == ind1_sil:
                for cond in condlist[0:2]:
                    minsamples[cond][ind1_sil, ind2_sil] = np.int(2048)

    # Minsamples[h0_llr]: LLR/Wilke's statistical test
    # ----------------------

    # fra-mle using llr
    # ******
    # Create likelihood function for each possible observation
    step = 0.01
    obs = np.arange(-2, 1 + 2 * step, step)  # Bin observations from -200:100
    likelihood = np.empty([len(obs), len(silent_truefrac)])
    for ind_obs, obs_ in enumerate(obs):
        for ind_hyp, hyp_ in enumerate(silent_truefrac):
            obs_in_range = np.where(
                np.abs(fra_calc[ind_hyp] - obs_) < step / 2)[0]
            p_obs = len(obs_in_range) / len(fra_calc[ind_hyp])
            likelihood[ind_obs, ind_hyp] = p_obs
    likelihood += 0.0001  # Set likelihoods away from 0 to avoid log(0) errors

    _cond = 'h0_llr_framle'
    for ind_sil, sil in enumerate(silent_truefrac):
        print('\nsil ' + str(sil))

        # Calc best guess based on last iter.
        if guess[_cond] == 2048:
            guess[_cond] = np.int(2048)
        else:
            exp_2 = np.ceil(
                np.log(minsamples[_cond][ind_sil - 1]) / np.log(2))
            guess[_cond] = np.int(2**(exp_2))

        # Calculate minsamples
        minsamples[_cond][ind_sil] = power_analysis_llr(
            fra_calc[ind_sil],
            likelihood,
            init_guess=guess[_cond],
            beta=conds_beta[_cond],
            sample_draws=sample_draws)

        guess[_cond] = minsamples[_cond][ind_sil]  # update guess

    # binary using llr analytical solution
    # ******
    silent_truefrac_nozeroes = silent_truefrac + 1 / 10000  # Move away from 0
    minsamples['h0_llr_binary'] = np.log(conds_beta['h0_llr_binary']) \
        / (np.log(1-silent_truefrac_nozeroes))

    # Minsamples[h0]: Monte Carlo simulation for H0 vs Ha (no model)
    # --------------------
    _conds = ['h0_fra', 'h0_binary']
    _matchconds = ['base', 'binary']

    h0_nsamples_templ = np.empty_like(silent_truefrac, dtype=np.ndarray)
    h0_nsamples = {cond: h0_nsamples_templ.copy()
                   for cond in _conds}

    for ind_sil, sil in enumerate(silent_truefrac):
        for ind_c, cond in enumerate(_conds):
            m_cond = _matchconds[ind_c]
            minsamples[cond][ind_sil] = minsamples[m_cond][0, ind_sil]

    # 2. Estimator bias and variance calculations
    ################################################
    # Compute bias and variance of corrected estimator
    # --------------------
    print('Calculating bias/variance...')
    est = Estimator(n_likelihood_points=n_true_silents_mle,
                    pr_dist_sil=PrDist(sp_stats.uniform),
                    pr_dist_nonsil=PrDist(sp_stats.uniform),
                    num_trials=100, failrate_low=0.2, failrate_high=0.8)
    est_hyp = est.hyp

    bias = np.empty(len(est_hyp))
    bias_framle = np.empty_like(bias)
    stdev = np.empty(len(est_hyp))
    stdev_framle = np.empty_like(stdev)

    for ind_sil, sil in enumerate(est_hyp):
        print(f'fra_mle: silent={sil}%', end='\r')
        _fra_dist = est.fra_dists[ind_sil]
        _llik_on_all = est.estimate(_fra_dist, plot_joint_likelihood=False,
                                    verbose=False)
        _mle_on_all = est.hyp[np.argmax(_llik_on_all)]

        _framle_dist = np.empty(len(_fra_dist))

        for ind_obs, _fra_obs in enumerate(_fra_dist):
            _lhood = est.estimate([_fra_obs],
                                  plot_joint_likelihood=False,
                                  verbose=False)
            _framle_dist[ind_obs] = est.hyp[np.argmax(_lhood)]

        bias[ind_sil] = np.mean(_fra_dist) - sil
        stdev[ind_sil] = np.std(_fra_dist)

        bias_framle[ind_sil] = _mle_on_all - sil
        stdev_framle[ind_sil] = np.std(_framle_dist)

    # 2. Make figure
    #####################################################
    try:
        plt.style.use('publication_ml')
    except FileNotFoundError:
        pass
    lw_ = 1  # universal linewidth argument passed to each plot

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(4)
    fig.set_figwidth(3.43)
    plt.rc('font', size=fontsize)

    # Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows=2,
                                 ncols=2,
                                 height_ratios=[1, 1],
                                 width_ratios=[1, 1],
                                 figure=fig, wspace=0.2,
                                 hspace=0.4)

    # Define colors to be used
    # color_1 = np.array([0.25, 0.55, 0.18])
    color_fra_palette = sns.diverging_palette(130, 30, l=45, s=90,
                                              center="dark", as_cmap=True)
    color_fra = color_fra_palette(0.1)
    color_binary = [0.2, 0.2, 0.2]
    color_llr_framle = cm_(0)
    color_llr_binary = cm_(0.95)

    # Top: FRA-MLE bias and variance
    # -----------------------------------------------
    ax_bias = fig.add_subplot(spec_all[0, 0])
    ax_bias.plot(est_hyp * 100, bias_framle * 100,
                 color=color_llr_framle, linewidth=lw_)
    ax_bias.plot(est_hyp * 100, bias * 100,
                 color=color_fra, linewidth=lw_)
    ax_bias.set_xlim([0, 50])
    ax_bias.set_xticks([0, 10, 20, 30, 40, 50])
    ax_bias.set_xlabel('true silent (%)')
    ax_bias.set_ylabel('estimator bias (%)')
    ax_bias.legend(['fra-mle', 'fra'])
    sns.despine(ax=ax_bias, offset=despine_offset)

    ax_var = fig.add_subplot(spec_all[0, 1])
    ax_var.plot(est_hyp * 100, stdev_framle * 100,
                color=color_llr_framle, linewidth=lw_)
    ax_var.plot(est_hyp * 100, stdev * 100,
                color=color_fra, linewidth=lw_)
    ax_var.set_xlim([0, 50])
    ax_var.set_xticks([0, 10, 20, 30, 40, 50])
    ax_var.set_ylim([0, 40])
    ax_var.set_xlabel('true silent (%)')
    ax_var.set_ylabel('estimator std. dev. (%)')
    ax_var.legend(['fra-mle', 'fra'])
    sns.despine(ax=ax_var, offset=despine_offset)

    # Bottom: Hypothesis testing
    # -----------------------------------------------
    # (FRA vs bin {discrim} | FRA-MLE vs bin {Ha vs H0})
    ax_power = fig.add_subplot(spec_all[1, :])
    # plot
    line_llr_framle = ax_power.plot(silent_truefrac * 100,
                                    minsamples['h0_llr_framle'],
                                    color=cm_(0),
                                    alpha=0.9,
                                    linewidth=lw_)
    line_llr_binary = ax_power.plot(silent_truefrac * 100,
                                    minsamples['h0_llr_binary'],
                                    color=cm_(0.95),
                                    alpha=0.9,
                                    linewidth=lw_)
    line_fra = ax_power.plot(silent_truefrac * 100,
                             minsamples['h0_fra'],
                             color=color_fra,
                             alpha=0.9, linewidth=lw_)
    line_bin = ax_power.plot(silent_truefrac * 100,
                             minsamples['h0_binary'],
                             color=color_binary,
                             alpha=0.9, linewidth=lw_)

    ax_power.set_xlim([0, 50])
    ax_power.set_ylim([0, 128])
    ax_power.set_xticks([0, 10, 20, 30, 40, 50])
    ax_power.set_yticks([0, 32, 64, 96, 128])
    ax_power.set_xlabel('true silent (%)')
    ax_power.set_ylabel('min samples')
    ax_power.set_title('Evidence against null',
                       color=[0.5, 0.5, 0.5],
                       fontweight='bold',
                       loc='left')
    sns.despine(ax=ax_power, offset=despine_offset)

    ax_power_inset = inset_axes(ax_power, width='60%', height='60%', loc=1)
    insline_llr_fmle = ax_power_inset.plot(silent_truefrac * 100,
                                           minsamples['h0_llr_framle'],
                                           color=color_llr_framle,
                                           alpha=0.9,
                                           linewidth=lw_)
    insline_llr_bin = ax_power_inset.plot(silent_truefrac * 100,
                                          minsamples['h0_llr_binary'],
                                          color=color_llr_binary,
                                          alpha=0.9,
                                          linewidth=lw_)
    insline_fra = ax_power_inset.plot(silent_truefrac * 100,
                                      minsamples['h0_fra'],
                                      color=color_fra,
                                      alpha=0.9, linewidth=lw_)
    insline_bin = ax_power_inset.plot(silent_truefrac * 100,
                                      minsamples['h0_binary'],
                                      color=color_binary,
                                      alpha=0.9, linewidth=lw_)

    ax_power_inset.set_xlim([15, 50])
    ax_power_inset.set_ylim([0, 30])
    ax_power_inset.set_xticks([15, 30, 45])
    ax_power_inset.set_yticks([0, 5, 10, 15, 20, 25, 30])
    ax_power_inset.legend([insline_fra[-1], insline_bin[-1],
                           insline_llr_fmle[-1], insline_llr_bin[-1]],
                          ['fra', 'binary', 'fra-mle (llr)', 'binary (llr)'],
                          frameon=False)
    mark_inset(ax_power,
               ax_power_inset,
               loc1=3,
               loc2=4,
               fc="none",
               ec="0.6",
               ls='--',
               lw=1)
    sns.despine(ax=ax_power_inset, offset=despine_offset)

    # ---------------------------------
    # Set tight layout and save
    fig.set_constrained_layout_pads(w_pad=0.001,
                                    h_pad=0.001,
                                    hspace=0.01,
                                    wspace=0.01)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    fig.savefig(path_f, bbox_inches='tight')

    return


def plot_figS4_bottom(n_true_silents_power=26,
                      n_true_silents_mle=500,
                      pr_dist_sil=PrDist(sp_stats.gamma,
                                         args={'a': 2,
                                               'scale': 1/5.8}),
                      pr_dist_nonsil=PrDist(sp_stats.gamma,
                                            args={'a': 3,
                                                  'scale': 1/5.8}),
                      frac_reduction=0.1,     
                      sample_draws=5000,
                      fontsize=8,
                      despine_offset={'left': 5},
                      figname='FigS4_bottom.pdf'):
    '''
    Plots the right side of Fig 5 (change in Pr between silent
    and nonsilent synapses)
    '''

    # 1. Estimator bias and variance calculations
    ################################################
    # Compute bias and variance of corrected estimator
    # --------------------
    print('Calculating bias/variance...')
    est = Estimator(n_likelihood_points=n_true_silents_mle,
                    pr_dist_sil=pr_dist_sil,
                    pr_dist_nonsil=pr_dist_nonsil,
                    frac_reduction=frac_reduction)
    est_hyp = est.hyp

    bias = np.empty(len(est_hyp))
    bias_framle = np.empty_like(bias)
    stdev = np.empty(len(est_hyp))
    stdev_framle = np.empty_like(stdev)

    for ind_sil, sil in enumerate(est_hyp):
        print(f'fra_mle: silent={sil*100}%', end='\r')
        _fra_dist = est.fra_dists[ind_sil]
        _llik_on_all = est.estimate(_fra_dist, plot_joint_likelihood=False,
                                    verbose=False)
        _mle_on_all = est.hyp[np.argmax(_llik_on_all)]

        _framle_dist = np.empty(len(_fra_dist))

        for ind_obs, _fra_obs in enumerate(_fra_dist):
            _lhood = est.estimate([_fra_obs],
                                  plot_joint_likelihood=False,
                                  verbose=False)
            _framle_dist[ind_obs] = est.hyp[np.argmax(_lhood)]

        bias[ind_sil] = np.mean(_fra_dist) - sil
        stdev[ind_sil] = np.std(_fra_dist)

        bias_framle[ind_sil] = _mle_on_all - sil
        stdev_framle[ind_sil] = np.std(_framle_dist)

    # 2. Make figure
    #####################################################
    try:
        plt.style.use('publication_ml')
    except FileNotFoundError:
        pass
    lw_ = 1  # universal linewidth argument passed to each plot

    fig = plt.figure(constrained_layout=True)
    fig.set_figheight(2)
    fig.set_figwidth(3.43)
    plt.rc('font', size=fontsize)

    # Define spec for entire fig
    spec_all = gridspec.GridSpec(nrows=1,
                                 ncols=2,
                                 figure=fig, wspace=0.2,
                                 hspace=0.4)

    # Define colors to be used
    # color_1 = np.array([0.25, 0.55, 0.18])
    color_fra_palette = sns.diverging_palette(130, 30, l=45, s=90,
                                              center="dark", as_cmap=True)
    color_fra = color_fra_palette(0.1)
    color_binary = [0.2, 0.2, 0.2]
    color_llr_framle = cm_(0)
    color_llr_binary = cm_(0.95)

    # Top: FRA-MLE bias and variance
    # -----------------------------------------------
    ax_bias = fig.add_subplot(spec_all[0, 0])
    ax_bias.plot(est_hyp * 100, bias_framle * 100,
                 color=color_llr_framle, linewidth=lw_)
    ax_bias.plot(est_hyp * 100, bias * 100,
                 color=color_fra, linewidth=lw_)
    ax_bias.set_xlim([0, 50])
    ax_bias.set_xticks([0, 10, 20, 30, 40, 50])
    ax_bias.set_xlabel('true silent (%)')
    ax_bias.set_ylabel('estimator bias (%)')
    ax_bias.legend(['fra-mle', 'fra'])
    sns.despine(ax=ax_bias, offset=despine_offset)

    ax_var = fig.add_subplot(spec_all[0, 1])
    ax_var.plot(est_hyp * 100, stdev_framle * 100,
                color=color_llr_framle, linewidth=lw_)
    ax_var.plot(est_hyp * 100, stdev * 100,
                color=color_fra, linewidth=lw_)
    ax_var.set_xlim([0, 50])
    ax_var.set_xticks([0, 10, 20, 30, 40, 50])
    ax_var.set_ylim([0, 40])
    ax_var.set_xlabel('true silent (%)')
    ax_var.set_ylabel('estimator std. dev. (%)')
    ax_var.legend(['fra-mle', 'fra'])
    sns.despine(ax=ax_var, offset=despine_offset)

    # ---------------------------------
    # Set tight layout and save
    fig.set_constrained_layout_pads(w_pad=0.001,
                                    h_pad=0.001,
                                    hspace=0.01,
                                    wspace=0.01)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    fig.savefig(path_f, bbox_inches='tight')
    plt.show()

    return


def plot_figS5(fname_data, ind_ex_data=1,
               est_simulations=10000,
               est_likelihood_pts=200,
               est_smooth_window=0.1,
               est_smooth_polyorder=3,
               n_redo_estimation=1,
               despine_offset={'left': 5},
               figname='FigS5.pdf'):

    """Plots figure S4 (MLE examples, and MLE on experimental silent estimates).
    """
    # Analyze experimental data
    # -------------------
    # Import data
    with h5py.File(fname_data) as f:
        fra_est_experim = f['sscalcs'][:] / 100

    # Analyze data
    if n_redo_estimation == 1:
        est = Estimator(n_simulations=est_simulations,
                        n_likelihood_points=est_likelihood_pts)
        joint_lhood_all = est.estimate(data=fra_est_experim,
                                       dtype='est',
                                       plot_joint_likelihood=False,
                                       smooth_window=est_smooth_window,
                                       smooth_polyorder=est_smooth_polyorder)
        joint_lhood_all /= np.max(joint_lhood_all)

    elif n_redo_estimation > 1:
        mle_all = np.empty(n_redo_estimation)
        for ind in range(n_redo_estimation):
            est = Estimator(n_simulations=est_simulations,
                            n_likelihood_points=est_likelihood_pts)
            joint_lhood_all = est.estimate(data=fra_est_experim,
                                           dtype='est',
                                           plot_joint_likelihood=False,
                                           smooth_window=est_smooth_window,
                                           smooth_polyorder=est_smooth_polyorder)
            joint_lhood_all /= np.max(joint_lhood_all)
            mle_all[ind] = est.hyp[np.argmax(joint_lhood_all)]

    joint_lhood_ex = est.estimate(data=[fra_est_experim[ind_ex_data]],
                                  dtype='est',
                                  plot_joint_likelihood=False,
                                  smooth_window=est_smooth_window,
                                  smooth_polyorder=est_smooth_polyorder)
    # Get max-likelihood estimate value for silent fraction
    ind_mle = np.argmax(joint_lhood_all)
    mle = est.hyp[ind_mle]

    # Plot figure
    # --------------------------------------------
    color_blue = sns.diverging_palette(
        240, 15, l=45, s=90, center="dark", as_cmap=True)(0)

    try:
        plt.style.use('pub_mbfl_spacious')
    except FileNotFoundError:
        pass

    fig = plt.figure(figsize=(6.85, 4))
    spec_all = gridspec.GridSpec(nrows=2, ncols=4,
                                 figure=fig)

    # Plot FRA-MLE estimator examples
    # ----------
    ax_mle_1obs = fig.add_subplot(spec_all[0, 0])
    ax_mle_20obs = fig.add_subplot(spec_all[0, 1])
    ax_mle_0sil = fig.add_subplot(spec_all[1, 0])
    ax_mle_25sil = fig.add_subplot(spec_all[1, 1])

    ex_silent = 0.2
    ind_ex_silent = np.argmin(np.abs(est.hyp-ex_silent))

    lhood_1obs = est.estimate([est.fra_dists[ind_ex_silent][0]],
                              plot_joint_likelihood=False)
    lhood_20obs = est.estimate(est.fra_dists[ind_ex_silent][0:20],
                               plot_joint_likelihood=False)

    ax_mle_1obs.plot(est.hyp*100, lhood_1obs, color=color_blue)
    ax_mle_1obs.plot([ex_silent*100, ex_silent*100],
                     [0, ax_mle_1obs.get_ylim()[1]], color='r',
                     linestyle='dashed', linewidth=1)
    ax_mle_1obs.set_ylim([0, ax_mle_1obs.get_ylim()[1]])
    ax_mle_1obs.set_xlabel('true silent (%)')
    ax_mle_1obs.set_ylabel('likelihood (norm.)')
    ax_mle_1obs.set_title('Single observation')
    ax_mle_1obs.set_xlim([0, 100])
    sns.despine(ax=ax_mle_1obs, offset=despine_offset)

    ax_mle_20obs.plot(est.hyp*100, lhood_20obs, color=color_blue)
    ax_mle_20obs.plot([ex_silent*100, ex_silent*100],
                      [0, ax_mle_20obs.get_ylim()[1]], color='r',
                      linestyle='dashed', linewidth=1)
    ax_mle_20obs.set_ylim([0, ax_mle_20obs.get_ylim()[1]])
    ax_mle_20obs.set_xlabel('true silent (%)')
    ax_mle_20obs.set_ylabel('likelihood (norm.)')
    ax_mle_20obs.set_title('n=20 observations')
    ax_mle_20obs.set_xlim([0, 100])
    sns.despine(ax=ax_mle_20obs, offset=despine_offset)

    # Construct FRA-MLE estimate distributions
    # -----------
    bins_ = np.arange(-100, 100, step=5)
    dist_exs = [0, 0.25]
    dist_axes = [ax_mle_0sil, ax_mle_25sil]

    for ind, dist_ex in enumerate(dist_exs):
        _ax = dist_axes[ind]
        ind_dist_ex = np.argmin(np.abs(est.hyp-dist_ex))

        _fra_dist = est.fra_dists[ind_dist_ex]
        _framle_dist = np.empty(len(_fra_dist))

        for ind_obs, _fra_obs in enumerate(_fra_dist):
            _lhood = est.estimate([_fra_obs],
                                  plot_joint_likelihood=False,
                                  verbose=False)
            _framle_dist[ind_obs] = est.hyp[np.argmax(_lhood)]

        _hist_framle = _ax.hist(_framle_dist*100, bins=bins_, histtype='step',
                                color=sns.xkcd_rgb['blue green'],
                                density=True, linewidth=0.8)
        _hist_fra = _ax.hist(_fra_dist*100, bins=bins_, histtype='step',
                             color='k', linewidth=0.8,
                             density=True)
        _ax.set_ylabel('pdf')
        _ax.set_xlabel('estimated silent (%)')
        _ax.legend(['FRA-MLE', 'FRA'])
        _ax.set_title(f'{dist_ex*100:.0f}% silent')
        _ax.set_xlim([-100, 100])
        sns.despine(ax=_ax, offset=despine_offset)

    # Plot experimental estimates using FRA-MLE
    # -------------

    ax_lhood_ex = fig.add_subplot(spec_all[0, 2])
    ax_lhood_all = fig.add_subplot(spec_all[0, 3])
    ax_mle_hist = fig.add_subplot(spec_all[1, 2:])

    ylim_max = 0.5
    ind_ylim_max = np.argmin(np.abs(ylim_max - est.hyp))

    # 1. likelihood function of single experiment
    ax_lhood_ex.plot(est.hyp[0:ind_ylim_max] * 100,
                     joint_lhood_ex[0:ind_ylim_max],
                     color=color_blue)
    ax_lhood_ex.set_xlabel('true silent (%)')
    ax_lhood_ex.set_ylabel('likelihood (norm.)')
    ax_lhood_ex.set_xlim([0, ax_lhood_ex.get_xlim()[1]])
    ax_lhood_ex.set_ylim([0, ax_lhood_ex.get_ylim()[1]])
    sns.despine(ax=ax_lhood_ex, offset=despine_offset)

    # 2. Joint likelihood function of all experiments
    ax_lhood_all.plot(est.hyp[0:ind_ylim_max] * 100,
                      joint_lhood_all[0:ind_ylim_max],
                      color=color_blue)
    ax_lhood_all.plot([mle*100, mle*100], [0, 1],
                      color=[0.85, 0.1, 0.2])
    ax_lhood_all.text(0.5, 0.6,
                      'Max-likelihood \nestimate: ' +
                      str(format(mle * 100, '.1f')) + '%',
                      color=[0.85, 0.1, 0.2],
                      transform=ax_lhood_all.transAxes,
                      horizontalalignment='left')
    ax_lhood_all.set_xlabel('true silent (%)')
    ax_lhood_all.set_ylabel('likelihood (norm.)')
    ax_lhood_all.set_xlim([0, ax_lhood_all.get_xlim()[1]])
    ax_lhood_all.set_ylim([0, ax_lhood_all.get_ylim()[1]])
    sns.despine(ax=ax_lhood_all, offset=despine_offset)

    # 3. Histogram of best estimate superimposed on distribution
    fra_dist_mle = est.fra_dists[ind_mle] * 100
    bins_hist = np.arange(-100, 100, 20)
    ax_mle_hist.hist(fra_est_experim*100,
                     bins=bins_hist,
                     weights=np.ones_like(fra_est_experim) /
                     len(fra_est_experim),
                     histtype='stepfilled',
                     edgecolor=[1, 1, 1],
                     facecolor=color_blue,
                     linewidth=0.8,
                     alpha=0.6)
    ax_mle_hist.hist(fra_dist_mle,
                     bins=bins_hist,
                     weights=np.ones_like(fra_dist_mle) /
                     len(fra_dist_mle),
                     histtype='step',
                     edgecolor=[0.85, 0.1, 0.2],
                     linewidth=0.8,
                     alpha=0.9)

    # 4. Print out the pval for KS test between the distributions
    stat, pval = sp_stats.ks_2samp(fra_dist_mle, fra_est_experim*100)
    print('KS test between exp. and MLE theor. dist: ' +
          str(format(pval, '.2f')))
    print('Mean of exp. distribution: ' +
          str(format(np.mean(fra_est_experim)*100, '.3f')))
    ax_mle_hist.set_xlim([-100, 100])
    ax_mle_hist.xaxis.set_ticks(np.linspace(-100, 100, 5))
    ax_mle_hist.spines['right'].set_visible(False)
    ax_mle_hist.spines['top'].set_visible(False)
    ax_mle_hist.yaxis.set_ticks_position('left')
    ax_mle_hist.xaxis.set_ticks_position('bottom')
    ax_mle_hist.set_xlabel('Estimated silent (%)')
    ax_mle_hist.set_ylabel('pdf')
    sns.despine(ax=ax_mle_hist, offset=despine_offset)

    spec_all.tight_layout(fig)

    path = os.path.join(os.getcwd(), 'figs')
    if not os.path.exists(path):
        os.makedirs(path)
    path_f = os.path.join(path, figname)

    fig.savefig(path_f, bbox_inches='tight')

    if n_redo_estimation > 1:
        fig2 = plt.figure(figsize=(3, 3))
        ax_multiest = fig2.add_subplot(1, 1, 1)
        ax_multiest.hist(mle_all, bins=20,
                         weights=np.ones_like(mle_all) /
                         len(mle_all),
                         histtype='step',
                         edgecolor=color_blue)
        ax_multiest.set_xlabel('estimate (% silent)')
        ax_multiest.set_ylabel('pdf')
        fig2.savefig('FigReview4.pdf')

        mle_all_stdev = np.std(mle_all)
        print(f'multiple estimation stdev: {mle_all_stdev}%')

    plt.show()

    return
