import array
import numbers
import warnings
from collections.abc import Iterable
from numbers import Integral, Real
import numpy as np
import scipy.sparse as sp
from scipy import linalg
from ..preprocessing import MultiLabelBinarizer
from ..utils import check_array, check_random_state
from ..utils import shuffle as util_shuffle
from ..utils._param_validation import Hidden, Interval, StrOptions, validate_params
from ..utils.random import sample_without_replacement
@validate_params({'mean': ['array-like', None], 'cov': [Interval(Real, 0, None, closed='left')], 'n_samples': [Interval(Integral, 1, None, closed='left')], 'n_features': [Interval(Integral, 1, None, closed='left')], 'n_classes': [Interval(Integral, 1, None, closed='left')], 'shuffle': ['boolean'], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_gaussian_quantiles(*, mean=None, cov=1.0, n_samples=100, n_features=2, n_classes=3, shuffle=True, random_state=None):
    """Generate isotropic Gaussian and label samples by quantile.

    This classification dataset is constructed by taking a multi-dimensional
    standard normal distribution and defining classes separated by nested
    concentric multi-dimensional spheres such that roughly equal numbers of
    samples are in each class (quantiles of the :math:`\\chi^2` distribution).

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    mean : array-like of shape (n_features,), default=None
        The mean of the multi-dimensional normal distribution.
        If None then use the origin (0, 0, ...).

    cov : float, default=1.0
        The covariance matrix will be this value times the unit matrix. This
        dataset only produces symmetric normal distributions.

    n_samples : int, default=100
        The total number of points equally divided among classes.

    n_features : int, default=2
        The number of features for each sample.

    n_classes : int, default=3
        The number of classes.

    shuffle : bool, default=True
        Shuffle the samples.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    y : ndarray of shape (n_samples,)
        The integer labels for quantile membership of each sample.

    Notes
    -----
    The dataset is from Zhu et al [1].

    References
    ----------
    .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.

    Examples
    --------
    >>> from sklearn.datasets import make_gaussian_quantiles
    >>> X, y = make_gaussian_quantiles(random_state=42)
    >>> X.shape
    (100, 2)
    >>> y.shape
    (100,)
    >>> list(y[:5])
    [2, 0, 1, 0, 2]
    """
    if n_samples < n_classes:
        raise ValueError('n_samples must be at least n_classes')
    generator = check_random_state(random_state)
    if mean is None:
        mean = np.zeros(n_features)
    else:
        mean = np.array(mean)
    X = generator.multivariate_normal(mean, cov * np.identity(n_features), (n_samples,))
    idx = np.argsort(np.sum((X - mean[np.newaxis, :]) ** 2, axis=1))
    X = X[idx, :]
    step = n_samples // n_classes
    y = np.hstack([np.repeat(np.arange(n_classes), step), np.repeat(n_classes - 1, n_samples - step * n_classes)])
    if shuffle:
        X, y = util_shuffle(X, y, random_state=generator)
    return (X, y)