import math
from warnings import warn
import numpy as np
from numpy.linalg import inv
from scipy import optimize, spatial
def ransac(data, model_class, min_samples, residual_threshold, is_data_valid=None, is_model_valid=None, max_trials=100, stop_sample_num=np.inf, stop_residuals_sum=0, stop_probability=1, rng=None, initial_inliers=None):
    """Fit a model to data with the RANSAC (random sample consensus) algorithm.

    RANSAC is an iterative algorithm for the robust estimation of parameters
    from a subset of inliers from the complete data set. Each iteration
    performs the following tasks:

    1. Select `min_samples` random samples from the original data and check
       whether the set of data is valid (see `is_data_valid`).
    2. Estimate a model to the random subset
       (`model_cls.estimate(*data[random_subset]`) and check whether the
       estimated model is valid (see `is_model_valid`).
    3. Classify all data as inliers or outliers by calculating the residuals
       to the estimated model (`model_cls.residuals(*data)`) - all data samples
       with residuals smaller than the `residual_threshold` are considered as
       inliers.
    4. Save estimated model as best model if number of inlier samples is
       maximal. In case the current estimated model has the same number of
       inliers, it is only considered as the best model if it has less sum of
       residuals.

    These steps are performed either a maximum number of times or until one of
    the special stop criteria are met. The final model is estimated using all
    inlier samples of the previously determined best model.

    Parameters
    ----------
    data : [list, tuple of] (N, ...) array
        Data set to which the model is fitted, where N is the number of data
        points and the remaining dimension are depending on model requirements.
        If the model class requires multiple input data arrays (e.g. source and
        destination coordinates of  ``skimage.transform.AffineTransform``),
        they can be optionally passed as tuple or list. Note, that in this case
        the functions ``estimate(*data)``, ``residuals(*data)``,
        ``is_model_valid(model, *random_data)`` and
        ``is_data_valid(*random_data)`` must all take each data array as
        separate arguments.
    model_class : object
        Object with the following object methods:

         * ``success = estimate(*data)``
         * ``residuals(*data)``

        where `success` indicates whether the model estimation succeeded
        (`True` or `None` for success, `False` for failure).
    min_samples : int in range (0, N)
        The minimum number of data points to fit a model to.
    residual_threshold : float larger than 0
        Maximum distance for a data point to be classified as an inlier.
    is_data_valid : function, optional
        This function is called with the randomly selected data before the
        model is fitted to it: `is_data_valid(*random_data)`.
    is_model_valid : function, optional
        This function is called with the estimated model and the randomly
        selected data: `is_model_valid(model, *random_data)`, .
    max_trials : int, optional
        Maximum number of iterations for random sample selection.
    stop_sample_num : int, optional
        Stop iteration if at least this number of inliers are found.
    stop_residuals_sum : float, optional
        Stop iteration if sum of residuals is less than or equal to this
        threshold.
    stop_probability : float in range [0, 1], optional
        RANSAC iteration stops if at least one outlier-free set of the
        training data is sampled with ``probability >= stop_probability``,
        depending on the current best model's inlier ratio and the number
        of trials. This requires to generate at least N samples (trials):

            N >= log(1 - probability) / log(1 - e**m)

        where the probability (confidence) is typically set to a high value
        such as 0.99, e is the current fraction of inliers w.r.t. the
        total number of samples, and m is the min_samples value.
    rng : {`numpy.random.Generator`, int}, optional
        Pseudo-random number generator.
        By default, a PCG64 generator is used (see :func:`numpy.random.default_rng`).
        If `rng` is an int, it is used to seed the generator.
    initial_inliers : array-like of bool, shape (N,), optional
        Initial samples selection for model estimation


    Returns
    -------
    model : object
        Best model with largest consensus set.
    inliers : (N,) array
        Boolean mask of inliers classified as ``True``.

    References
    ----------
    .. [1] "RANSAC", Wikipedia, https://en.wikipedia.org/wiki/RANSAC

    Examples
    --------

    Generate ellipse data without tilt and add noise:

    >>> t = np.linspace(0, 2 * np.pi, 50)
    >>> xc, yc = 20, 30
    >>> a, b = 5, 10
    >>> x = xc + a * np.cos(t)
    >>> y = yc + b * np.sin(t)
    >>> data = np.column_stack([x, y])
    >>> rng = np.random.default_rng(203560)  # do not copy this value
    >>> data += rng.normal(size=data.shape)

    Add some faulty data:

    >>> data[0] = (100, 100)
    >>> data[1] = (110, 120)
    >>> data[2] = (120, 130)
    >>> data[3] = (140, 130)

    Estimate ellipse model using all available data:

    >>> model = EllipseModel()
    >>> model.estimate(data)
    True
    >>> np.round(model.params)  # doctest: +SKIP
    array([ 72.,  75.,  77.,  14.,   1.])

    Estimate ellipse model using RANSAC:

    >>> ransac_model, inliers = ransac(data, EllipseModel, 20, 3, max_trials=50)
    >>> abs(np.round(ransac_model.params))  # doctest: +SKIP
    array([20., 30., 10.,  6.,  2.])
    >>> inliers  # doctest: +SKIP
    array([False, False, False, False,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True], dtype=bool)
    >>> sum(inliers) > 40
    True

    RANSAC can be used to robustly estimate a geometric
    transformation. In this section, we also show how to use a
    proportion of the total samples, rather than an absolute number.

    >>> from skimage.transform import SimilarityTransform
    >>> rng = np.random.default_rng()
    >>> src = 100 * rng.random((50, 2))
    >>> model0 = SimilarityTransform(scale=0.5, rotation=1,
    ...                              translation=(10, 20))
    >>> dst = model0(src)
    >>> dst[0] = (10000, 10000)
    >>> dst[1] = (-100, 100)
    >>> dst[2] = (50, 50)
    >>> ratio = 0.5  # use half of the samples
    >>> min_samples = int(ratio * len(src))
    >>> model, inliers = ransac(
    ...     (src, dst),
    ...     SimilarityTransform,
    ...     min_samples,
    ...     10,
    ...     initial_inliers=np.ones(len(src), dtype=bool),
    ... )  # doctest: +SKIP
    >>> inliers  # doctest: +SKIP
    array([False, False, False,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True])

    """
    best_inlier_num = 0
    best_inlier_residuals_sum = np.inf
    best_inliers = []
    validate_model = is_model_valid is not None
    validate_data = is_data_valid is not None
    rng = np.random.default_rng(rng)
    if not isinstance(data, tuple | list):
        data = (data,)
    num_samples = len(data[0])
    if not 0 < min_samples <= num_samples:
        raise ValueError(f'`min_samples` must be in range (0, {num_samples}]')
    if residual_threshold < 0:
        raise ValueError('`residual_threshold` must be greater than zero')
    if max_trials < 0:
        raise ValueError('`max_trials` must be greater than zero')
    if not 0 <= stop_probability <= 1:
        raise ValueError('`stop_probability` must be in range [0, 1]')
    if initial_inliers is not None and len(initial_inliers) != num_samples:
        raise ValueError(f"RANSAC received a vector of initial inliers (length {len(initial_inliers)}) that didn't match the number of samples ({num_samples}). The vector of initial inliers should have the same length as the number of samples and contain only True (this sample is an initial inlier) and False (this one isn't) values.")
    spl_idxs = initial_inliers if initial_inliers is not None else rng.choice(num_samples, min_samples, replace=False)
    model = model_class()
    num_trials = 0
    while num_trials < max_trials:
        num_trials += 1
        samples = [d[spl_idxs] for d in data]
        spl_idxs = rng.choice(num_samples, min_samples, replace=False)
        if validate_data and (not is_data_valid(*samples)):
            continue
        success = model.estimate(*samples)
        if success is not None and (not success):
            continue
        if validate_model and (not is_model_valid(model, *samples)):
            continue
        residuals = np.abs(model.residuals(*data))
        inliers = residuals < residual_threshold
        residuals_sum = residuals.dot(residuals)
        inliers_count = np.count_nonzero(inliers)
        if inliers_count > best_inlier_num or (inliers_count == best_inlier_num and residuals_sum < best_inlier_residuals_sum):
            best_inlier_num = inliers_count
            best_inlier_residuals_sum = residuals_sum
            best_inliers = inliers
            max_trials = min(max_trials, _dynamic_max_trials(best_inlier_num, num_samples, min_samples, stop_probability))
            if best_inlier_num >= stop_sample_num or best_inlier_residuals_sum <= stop_residuals_sum:
                break
    if any(best_inliers):
        data_inliers = [d[best_inliers] for d in data]
        model.estimate(*data_inliers)
        if validate_model and (not is_model_valid(model, *data_inliers)):
            warn('Estimated model is not valid. Try increasing max_trials.')
    else:
        model = None
        best_inliers = None
        warn('No inliers found. Model not fitted')
    return (model, best_inliers)