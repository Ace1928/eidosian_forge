from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def hdrboxplot(data, ncomp=2, alpha=None, threshold=0.95, bw=None, xdata=None, labels=None, ax=None, use_brute=False, seed=None):
    """
    High Density Region boxplot

    Parameters
    ----------
    data : sequence of ndarrays or 2-D ndarray
        The vectors of functions to create a functional boxplot from.  If a
        sequence of 1-D arrays, these should all be the same size.
        The first axis is the function index, the second axis the one along
        which the function is defined.  So ``data[0, :]`` is the first
        functional curve.
    ncomp : int, optional
        Number of components to use.  If None, returns the as many as the
        smaller of the number of rows or columns in data.
    alpha : list of floats between 0 and 1, optional
        Extra quantile values to compute. Default is None
    threshold : float between 0 and 1, optional
        Percentile threshold value for outliers detection. High value means
        a lower sensitivity to outliers. Default is `0.95`.
    bw : array_like or str, optional
        If an array, it is a fixed user-specified bandwidth. If `None`, set to
        `normal_reference`. If a string, should be one of:

            - normal_reference: normal reference rule of thumb (default)
            - cv_ml: cross validation maximum likelihood
            - cv_ls: cross validation least squares

    xdata : ndarray, optional
        The independent variable for the data. If not given, it is assumed to
        be an array of integers 0..N-1 with N the length of the vectors in
        `data`.
    labels : sequence of scalar or str, optional
        The labels or identifiers of the curves in `data`. If not given,
        outliers are labeled in the plot with array indices.
    ax : AxesSubplot, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    use_brute : bool
        Use the brute force optimizer instead of the default differential
        evolution to find the curves. Default is False.
    seed : {None, int, np.random.RandomState}
        Seed value to pass to scipy.optimize.differential_evolution. Can be an
        integer or RandomState instance. If None, then the default RandomState
        provided by np.random is used.

    Returns
    -------
    fig : Figure
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.
    hdr_res : HdrResults instance
        An `HdrResults` instance with the following attributes:

         - 'median', array. Median curve.
         - 'hdr_50', array. 50% quantile band. [sup, inf] curves
         - 'hdr_90', list of array. 90% quantile band. [sup, inf]
            curves.
         - 'extra_quantiles', list of array. Extra quantile band.
            [sup, inf] curves.
         - 'outliers', ndarray. Outlier curves.

    See Also
    --------
    banddepth, rainbowplot, fboxplot

    Notes
    -----
    The median curve is the curve with the highest probability on the reduced
    space of a Principal Component Analysis (PCA).

    Outliers are defined as curves that fall outside the band corresponding
    to the quantile given by `threshold`.

    The non-outlying region is defined as the band made up of all the
    non-outlying curves.

    Behind the scene, the dataset is represented as a matrix. Each line
    corresponding to a 1D curve. This matrix is then decomposed using Principal
    Components Analysis (PCA). This allows to represent the data using a finite
    number of modes, or components. This compression process allows to turn the
    functional representation into a scalar representation of the matrix. In
    other words, you can visualize each curve from its components. Each curve
    is thus a point in this reduced space. With 2 components, this is called a
    bivariate plot (2D plot).

    In this plot, if some points are adjacent (similar components), it means
    that back in the original space, the curves are similar. Then, finding the
    median curve means finding the higher density region (HDR) in the reduced
    space. Moreover, the more you get away from this HDR, the more the curve is
    unlikely to be similar to the other curves.

    Using a kernel smoothing technique, the probability density function (PDF)
    of the multivariate space can be recovered. From this PDF, it is possible
    to compute the density probability linked to the cluster of points and plot
    its contours.

    Finally, using these contours, the different quantiles can be extracted
    along with the median curve and the outliers.

    Steps to produce the HDR boxplot include:

    1. Compute a multivariate kernel density estimation
    2. Compute contour lines for quantiles 90%, 50% and `alpha` %
    3. Plot the bivariate plot
    4. Compute median curve along with quantiles and outliers curves.

    References
    ----------
    [1] R.J. Hyndman and H.L. Shang, "Rainbow Plots, Bagplots, and Boxplots for
        Functional Data", vol. 19, pp. 29-45, 2010.

    Examples
    --------
    Load the El Nino dataset.  Consists of 60 years worth of Pacific Ocean sea
    surface temperature data.

    >>> import matplotlib.pyplot as plt
    >>> import statsmodels.api as sm
    >>> data = sm.datasets.elnino.load()

    Create a functional boxplot.  We see that the years 1982-83 and 1997-98 are
    outliers; these are the years where El Nino (a climate pattern
    characterized by warming up of the sea surface and higher air pressures)
    occurred with unusual intensity.

    >>> fig = plt.figure()
    >>> ax = fig.add_subplot(111)
    >>> res = sm.graphics.hdrboxplot(data.raw_data[:, 1:],
    ...                              labels=data.raw_data[:, 0].astype(int),
    ...                              ax=ax)

    >>> ax.set_xlabel("Month of the year")
    >>> ax.set_ylabel("Sea surface temperature (C)")
    >>> ax.set_xticks(np.arange(13, step=3) - 1)
    >>> ax.set_xticklabels(["", "Mar", "Jun", "Sep", "Dec"])
    >>> ax.set_xlim([-0.2, 11.2])

    >>> plt.show()

    .. plot:: plots/graphics_functional_hdrboxplot.py
    """
    fig, ax = utils.create_mpl_ax(ax)
    if labels is None:
        if hasattr(data, 'index'):
            labels = data.index
        else:
            labels = np.arange(len(data))
    data = np.asarray(data)
    if xdata is None:
        xdata = np.arange(data.shape[1])
    n_samples, dim = data.shape
    pca = PCA(data, ncomp=ncomp)
    data_r = pca.factors
    ks_gaussian = KDEMultivariate(data_r, bw=bw, var_type='c' * data_r.shape[1])
    bounds = np.array([data_r.min(axis=0), data_r.max(axis=0)]).T
    if alpha is None:
        alpha = [threshold, 0.9, 0.5]
    else:
        alpha.extend([threshold, 0.9, 0.5])
        alpha = list(set(alpha))
    alpha.sort(reverse=True)
    n_quantiles = len(alpha)
    pdf_r = ks_gaussian.pdf(data_r).flatten()
    if NP_LT_123:
        pvalues = [np.percentile(pdf_r, (1 - alpha[i]) * 100, interpolation='linear') for i in range(n_quantiles)]
    else:
        pvalues = [np.percentile(pdf_r, (1 - alpha[i]) * 100, method='midpoint') for i in range(n_quantiles)]
    if have_de_optim and (not use_brute):
        median = differential_evolution(lambda x: -ks_gaussian.pdf(x), bounds=bounds, maxiter=5, seed=seed).x
    else:
        median = brute(lambda x: -ks_gaussian.pdf(x), ranges=bounds, finish=fmin)
    outliers_idx = np.where(pdf_r < pvalues[alpha.index(threshold)])[0]
    labels_outlier = [labels[i] for i in outliers_idx]
    outliers = data[outliers_idx]

    def _band_quantiles(band, use_brute=use_brute, seed=seed):
        """
        Find extreme curves for a quantile band.

        From the `band` of quantiles, the associated PDF extrema values
        are computed. If `min_alpha` is not provided (single quantile value),
        `max_pdf` is set to `1E6` in order not to constrain the problem on high
        values.

        An optimization is performed per component in order to find the min and
        max curves. This is done by comparing the PDF value of a given curve
        with the band PDF.

        Parameters
        ----------
        band : array_like
            alpha values ``(max_alpha, min_alpha)`` ex: ``[0.9, 0.5]``
        use_brute : bool
            Use the brute force optimizer instead of the default differential
            evolution to find the curves. Default is False.
        seed : {None, int, np.random.RandomState}
            Seed value to pass to scipy.optimize.differential_evolution. Can
            be an integer or RandomState instance. If None, then the default
            RandomState provided by np.random is used.


        Returns
        -------
        band_quantiles : list of 1-D array
            ``(max_quantile, min_quantile)`` (2, n_features)
        """
        min_pdf = pvalues[alpha.index(band[0])]
        try:
            max_pdf = pvalues[alpha.index(band[1])]
        except IndexError:
            max_pdf = 1000000.0
        band = [min_pdf, max_pdf]
        pool = Pool()
        data = zip(range(dim), itertools.repeat((band, pca, bounds, ks_gaussian, seed, use_brute)))
        band_quantiles = pool.map(_min_max_band, data)
        pool.terminate()
        pool.close()
        band_quantiles = list(zip(*band_quantiles))
        return band_quantiles
    extra_alpha = [i for i in alpha if 0.5 != i and 0.9 != i and (threshold != i)]
    if len(extra_alpha) > 0:
        extra_quantiles = []
        for x in extra_alpha:
            for y in _band_quantiles([x], use_brute=use_brute, seed=seed):
                extra_quantiles.append(y)
    else:
        extra_quantiles = []
    median = _inverse_transform(pca, median)[0]
    hdr_90 = _band_quantiles([0.9, 0.5], use_brute=use_brute, seed=seed)
    hdr_50 = _band_quantiles([0.5], use_brute=use_brute, seed=seed)
    hdr_res = HdrResults({'median': median, 'hdr_50': hdr_50, 'hdr_90': hdr_90, 'extra_quantiles': extra_quantiles, 'outliers': outliers, 'outliers_idx': outliers_idx})
    ax.plot(np.array([xdata] * n_samples).T, data.T, c='c', alpha=0.1, label=None)
    ax.plot(xdata, median, c='k', label='Median')
    fill_betweens = []
    fill_betweens.append(ax.fill_between(xdata, *hdr_50, color='gray', alpha=0.4, label='50% HDR'))
    fill_betweens.append(ax.fill_between(xdata, *hdr_90, color='gray', alpha=0.3, label='90% HDR'))
    if len(extra_quantiles) != 0:
        ax.plot(np.array([xdata] * len(extra_quantiles)).T, np.array(extra_quantiles).T, c='y', ls='-.', alpha=0.4, label='Extra quantiles')
    if len(outliers) != 0:
        for ii, outlier in enumerate(outliers):
            if labels_outlier is None:
                label = 'Outliers'
            else:
                label = str(labels_outlier[ii])
            ax.plot(xdata, outlier, ls='--', alpha=0.7, label=label)
    handles, labels = ax.get_legend_handles_labels()
    plt = _import_mpl()
    for label, fill_between in zip(['50% HDR', '90% HDR'], fill_betweens):
        p = plt.Rectangle((0, 0), 1, 1, fc=fill_between.get_facecolor()[0])
        handles.append(p)
        labels.append(label)
    by_label = dict(zip(labels, handles))
    if len(outliers) != 0:
        by_label.pop('Median')
        by_label.pop('50% HDR')
        by_label.pop('90% HDR')
    ax.legend(by_label.values(), by_label.keys(), loc='best')
    return (fig, hdr_res)