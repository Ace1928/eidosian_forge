import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
@pytest.mark.matplotlib
def test_plot_km(close_figures):
    if pdf_output:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf = PdfPages('test_survfunc.pdf')
    else:
        pdf = None
    sr1 = SurvfuncRight(ti1, st1)
    sr2 = SurvfuncRight(ti2, st2)
    fig = plot_survfunc(sr1)
    close_or_save(pdf, fig)
    fig = plot_survfunc(sr2)
    close_or_save(pdf, fig)
    fig = plot_survfunc([sr1, sr2])
    close_or_save(pdf, fig)
    gb = bmt.groupby('Group')
    sv = []
    for g in gb:
        s0 = SurvfuncRight(g[1]['T'], g[1]['Status'], title=g[0])
        sv.append(s0)
    fig = plot_survfunc(sv)
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    ha, lb = ax.get_legend_handles_labels()
    fig.legend([ha[k] for k in (0, 2, 4)], [lb[k] for k in (0, 2, 4)], loc='center right')
    close_or_save(pdf, fig)
    ii = bmt.Group == 'ALL'
    sf = SurvfuncRight(bmt.loc[ii, 'T'], bmt.loc[ii, 'Status'])
    fig = sf.plot()
    ax = fig.get_axes()[0]
    ax.set_position([0.1, 0.1, 0.64, 0.8])
    ha, lb = ax.get_legend_handles_labels()
    lcb, ucb = sf.simultaneous_cb(transform='log')
    plt.fill_between(sf.surv_times, lcb, ucb, color='lightgrey')
    lcb, ucb = sf.simultaneous_cb(transform='arcsin')
    plt.plot(sf.surv_times, lcb, color='darkgrey')
    plt.plot(sf.surv_times, ucb, color='darkgrey')
    plt.plot(sf.surv_times, sf.surv_prob - 2 * sf.surv_prob_se, color='red')
    plt.plot(sf.surv_times, sf.surv_prob + 2 * sf.surv_prob_se, color='red')
    plt.xlim(100, 600)
    close_or_save(pdf, fig)
    if pdf_output:
        pdf.close()