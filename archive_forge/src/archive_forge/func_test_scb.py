from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
@pytest.mark.matplotlib
def test_scb(self, close_figures):
    np.random.seed(473)
    n = 100
    x = np.random.normal(size=(n, 4))
    x[:, 0] = 1
    for fam_name in ('poisson', 'binomial', 'gaussian'):
        if fam_name == 'poisson':
            y = np.random.poisson(20, size=n)
            fam = sm.families.Poisson()
            true_mean = 20
            true_lp = np.log(20)
        elif fam_name == 'binomial':
            y = 1 * (np.random.uniform(size=n) < 0.5)
            fam = sm.families.Binomial()
            true_mean = 0.5
            true_lp = 0
        elif fam_name == 'gaussian':
            y = np.random.normal(size=n)
            fam = sm.families.Gaussian()
            true_mean = 0
            true_lp = 0
        model = sm.GLM(y, x, family=fam)
        result = model.fit()
        for linear in (False, True):
            true = true_lp if linear else true_mean
            values = {'const': 1, 'x2': 0}
            summaries = {'x3': np.mean}
            pred1, cb1, fvals1 = predict_functional(result, 'x1', values=values, summaries=summaries, linear=linear)
            pred2, cb2, fvals2 = predict_functional(result, 'x1', values=values, summaries=summaries, ci_method='simultaneous', linear=linear)
            plt.clf()
            fig = plt.figure()
            ax = plt.axes([0.1, 0.1, 0.58, 0.8])
            plt.plot(fvals1, pred1, '-', color='black', label='Estimate')
            plt.plot(fvals1, true * np.ones(len(pred1)), '-', color='purple', label='Truth')
            plt.plot(fvals1, cb1[:, 0], color='blue', label='Pointwise CB')
            plt.plot(fvals1, cb1[:, 1], color='blue')
            plt.plot(fvals2, cb2[:, 0], color='green', label='Simultaneous CB')
            plt.plot(fvals2, cb2[:, 1], color='green')
            ha, lb = ax.get_legend_handles_labels()
            leg = plt.figlegend(ha, lb, loc='center right')
            leg.draw_frame(False)
            plt.xlabel('Focus variable', size=15)
            if linear:
                plt.ylabel('Linear predictor', size=15)
            else:
                plt.ylabel('Fitted mean', size=15)
            plt.title('%s family prediction' % fam_name.capitalize())
            self.close_or_save(fig)