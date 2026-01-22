import codecs
import logging
import os
import pickle
import re
import shutil
import tarfile
from contextlib import suppress
import joblib
import numpy as np
import scipy.sparse as sp
from .. import preprocessing
from ..feature_extraction.text import CountVectorizer
from ..utils import Bunch, check_random_state
from ..utils._param_validation import StrOptions, validate_params
from . import get_data_home, load_files
from ._base import (
@validate_params({'data_home': [str, os.PathLike, None], 'subset': [StrOptions({'train', 'test', 'all'})], 'categories': ['array-like', None], 'shuffle': ['boolean'], 'random_state': ['random_state'], 'remove': [tuple], 'download_if_missing': ['boolean'], 'return_X_y': ['boolean']}, prefer_skip_nested_validation=True)
def fetch_20newsgroups(*, data_home=None, subset='train', categories=None, shuffle=True, random_state=42, remove=(), download_if_missing=True, return_X_y=False):
    """Load the filenames and data from the 20 newsgroups dataset (classification).

    Download it if necessary.

    =================   ==========
    Classes                     20
    Samples total            18846
    Dimensionality               1
    Features                  text
    =================   ==========

    Read more in the :ref:`User Guide <20newsgroups_dataset>`.

    Parameters
    ----------
    data_home : str or path-like, default=None
        Specify a download and cache folder for the datasets. If None,
        all scikit-learn data is stored in '~/scikit_learn_data' subfolders.

    subset : {'train', 'test', 'all'}, default='train'
        Select the dataset to load: 'train' for the training set, 'test'
        for the test set, 'all' for both, with shuffled ordering.

    categories : array-like, dtype=str, default=None
        If None (default), load all the categories.
        If not None, list of category names to load (other categories
        ignored).

    shuffle : bool, default=True
        Whether or not to shuffle the data: might be important for models that
        make the assumption that the samples are independent and identically
        distributed (i.i.d.), such as stochastic gradient descent.

    random_state : int, RandomState instance or None, default=42
        Determines random number generation for dataset shuffling. Pass an int
        for reproducible output across multiple function calls.
        See :term:`Glossary <random_state>`.

    remove : tuple, default=()
        May contain any subset of ('headers', 'footers', 'quotes'). Each of
        these are kinds of text that will be detected and removed from the
        newsgroup posts, preventing classifiers from overfitting on
        metadata.

        'headers' removes newsgroup headers, 'footers' removes blocks at the
        ends of posts that look like signatures, and 'quotes' removes lines
        that appear to be quoting another post.

        'headers' follows an exact standard; the other filters are not always
        correct.

    download_if_missing : bool, default=True
        If False, raise an OSError if the data is not locally available
        instead of trying to download the data from the source site.

    return_X_y : bool, default=False
        If True, returns `(data.data, data.target)` instead of a Bunch
        object.

        .. versionadded:: 0.22

    Returns
    -------
    bunch : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        data : list of shape (n_samples,)
            The data list to learn.
        target: ndarray of shape (n_samples,)
            The target labels.
        filenames: list of shape (n_samples,)
            The path to the location of the data.
        DESCR: str
            The full description of the dataset.
        target_names: list of shape (n_classes,)
            The names of target classes.

    (data, target) : tuple if `return_X_y=True`
        A tuple of two ndarrays. The first contains a 2D array of shape
        (n_samples, n_classes) with each row representing one sample and each
        column representing the features. The second array of shape
        (n_samples,) contains the target samples.

        .. versionadded:: 0.22
    """
    data_home = get_data_home(data_home=data_home)
    cache_path = _pkl_filepath(data_home, CACHE_NAME)
    twenty_home = os.path.join(data_home, '20news_home')
    cache = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                compressed_content = f.read()
            uncompressed_content = codecs.decode(compressed_content, 'zlib_codec')
            cache = pickle.loads(uncompressed_content)
        except Exception as e:
            print(80 * '_')
            print('Cache loading failed')
            print(80 * '_')
            print(e)
    if cache is None:
        if download_if_missing:
            logger.info('Downloading 20news dataset. This may take a few minutes.')
            cache = _download_20newsgroups(target_dir=twenty_home, cache_path=cache_path)
        else:
            raise OSError('20Newsgroups dataset not found')
    if subset in ('train', 'test'):
        data = cache[subset]
    elif subset == 'all':
        data_lst = list()
        target = list()
        filenames = list()
        for subset in ('train', 'test'):
            data = cache[subset]
            data_lst.extend(data.data)
            target.extend(data.target)
            filenames.extend(data.filenames)
        data.data = data_lst
        data.target = np.array(target)
        data.filenames = np.array(filenames)
    fdescr = load_descr('twenty_newsgroups.rst')
    data.DESCR = fdescr
    if 'headers' in remove:
        data.data = [strip_newsgroup_header(text) for text in data.data]
    if 'footers' in remove:
        data.data = [strip_newsgroup_footer(text) for text in data.data]
    if 'quotes' in remove:
        data.data = [strip_newsgroup_quoting(text) for text in data.data]
    if categories is not None:
        labels = [(data.target_names.index(cat), cat) for cat in categories]
        labels.sort()
        labels, categories = zip(*labels)
        mask = np.isin(data.target, labels)
        data.filenames = data.filenames[mask]
        data.target = data.target[mask]
        data.target = np.searchsorted(labels, data.target)
        data.target_names = list(categories)
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[mask]
        data.data = data_lst.tolist()
    if shuffle:
        random_state = check_random_state(random_state)
        indices = np.arange(data.target.shape[0])
        random_state.shuffle(indices)
        data.filenames = data.filenames[indices]
        data.target = data.target[indices]
        data_lst = np.array(data.data, dtype=object)
        data_lst = data_lst[indices]
        data.data = data_lst.tolist()
    if return_X_y:
        return (data.data, data.target)
    return data