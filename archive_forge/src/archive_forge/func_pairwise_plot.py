import itertools
from pyomo.common.dependencies import (
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.common.dependencies.scipy import stats
imports_available = (
def pairwise_plot(theta_values, theta_star=None, alpha=None, distributions=[], axis_limits=None, title=None, add_obj_contour=True, add_legend=True, filename=None):
    """
    Plot pairwise relationship for theta values, and optionally alpha-level
    confidence intervals and objective value contours

    Parameters
    ----------
    theta_values: DataFrame or tuple

        * If theta_values is a DataFrame, then it contains one column for each theta variable
          and (optionally) an objective value column ('obj') and columns that contains
          Boolean results from confidence interval tests (labeled using the alpha value).
          Each row is a sample.

            * Theta variables can be computed from ``theta_est_bootstrap``,
              ``theta_est_leaveNout``, and  ``leaveNout_bootstrap_test``.
            * The objective value can be computed using the ``likelihood_ratio_test``.
            * Results from confidence interval tests can be computed using the
              ``leaveNout_bootstrap_test``, ``likelihood_ratio_test``, and
              ``confidence_region_test``.

        * If theta_values is a tuple, then it contains a mean, covariance, and number
          of samples (mean, cov, n) where mean is a dictionary or Series
          (indexed by variable name), covariance is a DataFrame (indexed by
          variable name, one column per variable name), and n is an integer.
          The mean and covariance are used to create a multivariate normal
          sample of n theta values. The covariance can be computed using
          ``theta_est(calc_cov=True)``.

    theta_star: dict or Series, optional
        Estimated value of theta.  The dictionary or Series is indexed by variable name.
        Theta_star is used to slice higher dimensional contour intervals in 2D
    alpha: float, optional
        Confidence interval value, if an alpha value is given and the
        distributions list is empty, the data will be filtered by True/False
        values using the column name whose value equals alpha (see results from
        ``leaveNout_bootstrap_test``, ``likelihood_ratio_test``, and
        ``confidence_region_test``)
    distributions: list of strings, optional
        Statistical distribution used to define a confidence region,
        options = 'MVN' for multivariate_normal, 'KDE' for gaussian_kde, and
        'Rect' for rectangular.
        Confidence interval is a 2D slice, using linear interpolation at theta_star.
    axis_limits: dict, optional
        Axis limits in the format {variable: [min, max]}
    title: string, optional
        Plot title
    add_obj_contour: bool, optional
        Add a contour plot using the column 'obj' in theta_values.
        Contour plot is a 2D slice, using linear interpolation at theta_star.
    add_legend: bool, optional
        Add a legend to the plot
    filename: string, optional
        Filename used to save the figure
    """
    assert isinstance(theta_values, (pd.DataFrame, tuple))
    assert isinstance(theta_star, (type(None), dict, pd.Series, pd.DataFrame))
    assert isinstance(alpha, (type(None), int, float))
    assert isinstance(distributions, list)
    assert set(distributions).issubset(set(['MVN', 'KDE', 'Rect']))
    assert isinstance(axis_limits, (type(None), dict))
    assert isinstance(title, (type(None), str))
    assert isinstance(add_obj_contour, bool)
    assert isinstance(filename, (type(None), str))
    if isinstance(theta_values, tuple):
        assert len(theta_values) == 3
        mean = theta_values[0]
        cov = theta_values[1]
        n = theta_values[2]
        if isinstance(mean, dict):
            mean = pd.Series(mean)
        theta_names = mean.index
        mvn_dist = stats.multivariate_normal(mean, cov)
        theta_values = pd.DataFrame(mvn_dist.rvs(n, random_state=1), columns=theta_names)
    assert theta_values.shape[0] > 0
    if isinstance(theta_star, dict):
        theta_star = pd.Series(theta_star)
    if isinstance(theta_star, pd.DataFrame):
        theta_star = theta_star.loc[0, :]
    theta_names = [col for col in theta_values.columns if col not in ['obj'] and (not isinstance(col, float)) and (not isinstance(col, int))]
    if alpha in theta_values.columns and len(distributions) == 0:
        thetas = theta_values.loc[theta_values[alpha] == True, theta_names]
    else:
        thetas = theta_values[theta_names]
    if theta_star is not None:
        theta_star = theta_star[theta_names]
    legend_elements = []
    g = sns.PairGrid(thetas)
    if check_min_version(sns, '0.11'):
        g.map_diag(sns.histplot)
    else:
        g.map_diag(sns.distplot, kde=False, hist=True, norm_hist=False)
    if 'obj' in theta_values.columns and add_obj_contour:
        g.map_offdiag(_add_obj_contour, columns=theta_names, data=theta_values, theta_star=theta_star)
    g.map_offdiag(plt.scatter, s=10)
    legend_elements.append(matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='thetas', markerfacecolor='cadetblue', markersize=5))
    if theta_star is not None:
        g.map_offdiag(_add_scatter, color='k', columns=theta_names, theta_star=theta_star)
        legend_elements.append(matplotlib.lines.Line2D([0], [0], marker='o', color='w', label='theta*', markerfacecolor='k', markersize=6))
    colors = ['r', 'mediumblue', 'darkgray']
    if alpha is not None and len(distributions) > 0:
        if theta_star is None:
            print('theta_star is not defined, confidence region slice will be \n                  plotted at the mean value of theta')
            theta_star = thetas.mean()
        mvn_dist = None
        kde_dist = None
        for i, dist in enumerate(distributions):
            if dist == 'Rect':
                lb, ub = fit_rect_dist(thetas, alpha)
                g.map_offdiag(_add_rectangle_CI, color=colors[i], columns=theta_names, lower_bound=lb, upper_bound=ub)
                legend_elements.append(matplotlib.lines.Line2D([0], [0], color=colors[i], lw=1, label=dist))
            elif dist == 'MVN':
                mvn_dist = fit_mvn_dist(thetas)
                Z = mvn_dist.pdf(thetas)
                score = stats.scoreatpercentile(Z, (1 - alpha) * 100)
                g.map_offdiag(_add_scipy_dist_CI, color=colors[i], columns=theta_names, ncells=100, alpha=score, dist=mvn_dist, theta_star=theta_star)
                legend_elements.append(matplotlib.lines.Line2D([0], [0], color=colors[i], lw=1, label=dist))
            elif dist == 'KDE':
                kde_dist = fit_kde_dist(thetas)
                Z = kde_dist.pdf(thetas.transpose())
                score = stats.scoreatpercentile(Z, (1 - alpha) * 100)
                g.map_offdiag(_add_scipy_dist_CI, color=colors[i], columns=theta_names, ncells=100, alpha=score, dist=kde_dist, theta_star=theta_star)
                legend_elements.append(matplotlib.lines.Line2D([0], [0], color=colors[i], lw=1, label=dist))
    _set_axis_limits(g, axis_limits, thetas, theta_star)
    for ax in g.axes.flatten():
        ax.ticklabel_format(style='sci', scilimits=(-2, 2), axis='both')
        if add_legend:
            xvar, yvar, loc = _get_variables(ax, theta_names)
            if loc == (len(theta_names) - 1, 0):
                ax.legend(handles=legend_elements, loc='best', prop={'size': 8})
    if title:
        g.fig.subplots_adjust(top=0.9)
        g.fig.suptitle(title)
    lower_triangle_only = False
    if lower_triangle_only:
        for ax in g.axes.flatten():
            xvar, yvar, (xloc, yloc) = _get_variables(ax, theta_names)
            if xloc < yloc:
                ax.remove()
                ax.set_xlabel(xvar)
                ax.set_ylabel(yvar)
                fig = plt.figure()
                ax.figure = fig
                fig.axes.append(ax)
                fig.add_axes(ax)
                f, dummy = plt.subplots()
                bbox = dummy.get_position()
                ax.set_position(bbox)
                dummy.remove()
                plt.close(f)
                ax.tick_params(reset=True)
                if add_legend:
                    ax.legend(handles=legend_elements, loc='best', prop={'size': 8})
        plt.close(g.fig)
    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()