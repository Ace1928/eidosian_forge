from statsmodels.sandbox.predict_functional import predict_functional
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
@pytest.mark.matplotlib
def test_lm_contrast(self, close_figures):
    np.random.seed(542)
    n = 200
    x1 = np.random.normal(size=n)
    x2 = np.random.normal(size=n)
    x3 = np.random.normal(size=n)
    y = x1 + 2 * x2 + x3 - x1 * x2 + x2 * x3 + np.random.normal(size=n)
    df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'x3': x3})
    fml = 'y ~ x1 + x2 + x3 + x1*x2 + x2*x3'
    model = sm.OLS.from_formula(fml, data=df)
    result = model.fit()
    values = {'x2': 1, 'x3': 1}
    values2 = {'x2': 0, 'x3': 0}
    pr, cb, fvals = predict_functional(result, 'x1', values=values, values2=values2, ci_method='scheffe')
    plt.clf()
    fig = plt.figure()
    ax = plt.axes([0.1, 0.1, 0.67, 0.8])
    plt.plot(fvals, pr, '-', label='Estimate', color='orange', lw=4)
    plt.plot(fvals, 4 - fvals, '-', label='Truth', color='lime', lw=4)
    plt.fill_between(fvals, cb[:, 0], cb[:, 1], color='grey')
    ha, lb = ax.get_legend_handles_labels()
    leg = plt.figlegend(ha, lb, loc='center right')
    leg.draw_frame(False)
    plt.xlabel('Focus variable', size=15)
    plt.ylabel('Mean contrast', size=15)
    plt.title('Linear model contrast')
    self.close_or_save(fig)