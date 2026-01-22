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
@validate_params({'n_samples': [Interval(Integral, 1, None, closed='left')], 'n_features': [Interval(Integral, 1, None, closed='left')], 'n_classes': [Interval(Integral, 1, None, closed='left')], 'n_labels': [Interval(Integral, 0, None, closed='left')], 'length': [Interval(Integral, 1, None, closed='left')], 'allow_unlabeled': ['boolean'], 'sparse': ['boolean'], 'return_indicator': [StrOptions({'dense', 'sparse'}), 'boolean'], 'return_distributions': ['boolean'], 'random_state': ['random_state']}, prefer_skip_nested_validation=True)
def make_multilabel_classification(n_samples=100, n_features=20, *, n_classes=5, n_labels=2, length=50, allow_unlabeled=True, sparse=False, return_indicator='dense', return_distributions=False, random_state=None):
    """Generate a random multilabel classification problem.

    For each sample, the generative process is:
        - pick the number of labels: n ~ Poisson(n_labels)
        - n times, choose a class c: c ~ Multinomial(theta)
        - pick the document length: k ~ Poisson(length)
        - k times, choose a word: w ~ Multinomial(theta_c)

    In the above process, rejection sampling is used to make sure that
    n is never zero or more than `n_classes`, and that the document length
    is never zero. Likewise, we reject classes which have already been chosen.

    For an example of usage, see
    :ref:`sphx_glr_auto_examples_datasets_plot_random_multilabel_dataset.py`.

    Read more in the :ref:`User Guide <sample_generators>`.

    Parameters
    ----------
    n_samples : int, default=100
        The number of samples.

    n_features : int, default=20
        The total number of features.

    n_classes : int, default=5
        The number of classes of the classification problem.

    n_labels : int, default=2
        The average number of labels per instance. More precisely, the number
        of labels per sample is drawn from a Poisson distribution with
        ``n_labels`` as its expected value, but samples are bounded (using
        rejection sampling) by ``n_classes``, and must be nonzero if
        ``allow_unlabeled`` is False.

    length : int, default=50
        The sum of the features (number of words if documents) is drawn from
        a Poisson distribution with this expected value.

    allow_unlabeled : bool, default=True
        If ``True``, some instances might not belong to any class.

    sparse : bool, default=False
        If ``True``, return a sparse feature matrix.

        .. versionadded:: 0.17
           parameter to allow *sparse* output.

    return_indicator : {'dense', 'sparse'} or False, default='dense'
        If ``'dense'`` return ``Y`` in the dense binary indicator format. If
        ``'sparse'`` return ``Y`` in the sparse binary indicator format.
        ``False`` returns a list of lists of labels.

    return_distributions : bool, default=False
        If ``True``, return the prior class probability and conditional
        probabilities of features given classes, from which the data was
        drawn.

    random_state : int, RandomState instance or None, default=None
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated samples.

    Y : {ndarray, sparse matrix} of shape (n_samples, n_classes)
        The label sets. Sparse matrix should be of CSR format.

    p_c : ndarray of shape (n_classes,)
        The probability of each class being drawn. Only returned if
        ``return_distributions=True``.

    p_w_c : ndarray of shape (n_features, n_classes)
        The probability of each feature being drawn given each class.
        Only returned if ``return_distributions=True``.

    Examples
    --------
    >>> from sklearn.datasets import make_multilabel_classification
    >>> X, y = make_multilabel_classification(n_labels=3, random_state=42)
    >>> X.shape
    (100, 20)
    >>> y.shape
    (100, 5)
    >>> list(y[:3])
    [array([1, 1, 0, 1, 0]), array([0, 1, 1, 1, 0]), array([0, 1, 0, 0, 0])]
    """
    generator = check_random_state(random_state)
    p_c = generator.uniform(size=n_classes)
    p_c /= p_c.sum()
    cumulative_p_c = np.cumsum(p_c)
    p_w_c = generator.uniform(size=(n_features, n_classes))
    p_w_c /= np.sum(p_w_c, axis=0)

    def sample_example():
        _, n_classes = p_w_c.shape
        y_size = n_classes + 1
        while not allow_unlabeled and y_size == 0 or y_size > n_classes:
            y_size = generator.poisson(n_labels)
        y = set()
        while len(y) != y_size:
            c = np.searchsorted(cumulative_p_c, generator.uniform(size=y_size - len(y)))
            y.update(c)
        y = list(y)
        n_words = 0
        while n_words == 0:
            n_words = generator.poisson(length)
        if len(y) == 0:
            words = generator.randint(n_features, size=n_words)
            return (words, y)
        cumulative_p_w_sample = p_w_c.take(y, axis=1).sum(axis=1).cumsum()
        cumulative_p_w_sample /= cumulative_p_w_sample[-1]
        words = np.searchsorted(cumulative_p_w_sample, generator.uniform(size=n_words))
        return (words, y)
    X_indices = array.array('i')
    X_indptr = array.array('i', [0])
    Y = []
    for i in range(n_samples):
        words, y = sample_example()
        X_indices.extend(words)
        X_indptr.append(len(X_indices))
        Y.append(y)
    X_data = np.ones(len(X_indices), dtype=np.float64)
    X = sp.csr_matrix((X_data, X_indices, X_indptr), shape=(n_samples, n_features))
    X.sum_duplicates()
    if not sparse:
        X = X.toarray()
    if return_indicator in (True, 'sparse', 'dense'):
        lb = MultiLabelBinarizer(sparse_output=return_indicator == 'sparse')
        Y = lb.fit([range(n_classes)]).transform(Y)
    if return_distributions:
        return (X, Y, p_c, p_w_c)
    return (X, Y)