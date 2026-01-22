import numpy as np
from scipy import optimize
from scipy.stats import chi2, skew, kurtosis
from statsmodels.base.optimizer import _fit_newton
import itertools
from statsmodels.graphics import utils
def mv_mean_contour(self, mu1_low, mu1_upp, mu2_low, mu2_upp, step1, step2, levs=(0.001, 0.01, 0.05, 0.1, 0.2), var1_name=None, var2_name=None, plot_dta=False):
    """
        Creates a confidence region plot for the mean of bivariate data

        Parameters
        ----------
        m1_low : float
            Minimum value of the mean for variable 1
        m1_upp : float
            Maximum value of the mean for variable 1
        mu2_low : float
            Minimum value of the mean for variable 2
        mu2_upp : float
            Maximum value of the mean for variable 2
        step1 : float
            Increment of evaluations for variable 1
        step2 : float
            Increment of evaluations for variable 2
        levs : list
            Levels to be drawn on the contour plot.
            Default =  (.001, .01, .05, .1, .2)
        plot_dta : bool
            If True, makes a scatter plot of the data on
            top of the contour plot. Defaultis False.
        var1_name : str
            Name of variable 1 to be plotted on the x-axis
        var2_name : str
            Name of variable 2 to be plotted on the y-axis

        Notes
        -----
        The smaller the step size, the more accurate the intervals
        will be

        If the function returns optimization failed, consider narrowing
        the boundaries of the plot

        Examples
        --------
        >>> import statsmodels.api as sm
        >>> two_rvs = np.random.standard_normal((20,2))
        >>> el_analysis = sm.emplike.DescStat(two_rvs)
        >>> contourp = el_analysis.mv_mean_contour(-2, 2, -2, 2, .1, .1)
        >>> contourp.show()
        """
    if self.endog.shape[1] != 2:
        raise ValueError('Data must contain exactly two variables')
    fig, ax = utils.create_mpl_ax()
    if var2_name is None:
        ax.set_ylabel('Variable 2')
    else:
        ax.set_ylabel(var2_name)
    if var1_name is None:
        ax.set_xlabel('Variable 1')
    else:
        ax.set_xlabel(var1_name)
    x = np.arange(mu1_low, mu1_upp, step1)
    y = np.arange(mu2_low, mu2_upp, step2)
    pairs = itertools.product(x, y)
    z = []
    for i in pairs:
        z.append(self.mv_test_mean(np.asarray(i))[0])
    X, Y = np.meshgrid(x, y)
    z = np.asarray(z)
    z = z.reshape(X.shape[1], Y.shape[0])
    ax.contour(x, y, z.T, levels=levs)
    if plot_dta:
        ax.plot(self.endog[:, 0], self.endog[:, 1], 'bo')
    return fig