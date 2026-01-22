from contextlib import contextmanager  # noqa E402
from copy import deepcopy
import logging
import sys
import os
from collections import OrderedDict, defaultdict
from six import iteritems, string_types, integer_types
import warnings
import numpy as np
import ctypes
import platform
import tempfile
import shutil
import json
from enum import Enum
from operator import itemgetter
import threading
import scipy.sparse
from .plot_helpers import save_plot_file, try_plot_offline, OfflineMetricVisualizer
from . import _catboost
from .metrics import BuiltinMetric
def sample_gaussian_process(X, y, eval_set=None, cat_features=None, text_features=None, embedding_features=None, random_seed=None, samples=10, posterior_iterations=900, prior_iterations=100, learning_rate=0.1, depth=6, sigma=0.1, delta=0, random_strength=0.1, random_score_type='Gumbel', eps=0.0001, verbose=False):
    """
    Implementation of Gaussian process sampling (Kernel Gradient Boosting/Algorithm 4) from "Gradient Boosting Performs Gaussian Process Inference" https://arxiv.org/abs/2206.05608
    Produces samples from posterior GP with prior assumption f ~ GP(0, sigma ** 2 K + delta ** 2 I)

    Parameters
    ----------
    X : list or numpy.ndarray or pandas.DataFrame or pandas.Series or catboost.FeaturesData
        If catboost.FeaturesData it must be 2 dimensional Feature matrix
        Must be non-empty (contain > 0 objects)
    y : list or numpy.ndarray or pandas.DataFrame or pandas.Series
        Labels of the training data.
        Must be a single-dimensional array with numerical values.
    eval_set : catboost.Pool or list of catboost.Pool or tuple (X, y) or list [(X, y)], optional (default=None)
        Validation dataset or datasets for metrics calculation and possibly early stopping in posterior training.
    cat_features : list or numpy.ndarray, optional (default=None)
        If not None, giving the list of Categ columns indices.
        Use only if X is not catboost.FeaturesData
    text_features : list or numpy.ndarray, optional (default=None)
        If not none, giving the list of Text columns indices.
        Use only if X is not catboost.FeaturesData
    embedding_features : list or numpy.ndarray, optional (default=None)
        If not none, giving the list of Embedding columns indices.
        Use only if X is not catboost.FeaturesData
    random_seed : int, [default=None]
        Random number seed.
        If None, 0 is used.
        range: [0,+inf)
    samples : int, [default=10]
        Number of Monte-Carlo samples from GP posterior. Controls how many models this function will return.
        range: [1,+inf)
    posterior_iterations : int, [default=900]
        Max count of trees for posterior sampling step.
        range: [1,+inf)
    prior_iterations : int, [default=100]
        Max count of trees for prior sampling step.
        range: [1,+inf]
    learning_rate : float, [default=0.1]
        Step size shrinkage used in update to prevent overfitting.
        range: (0,1]
    depth : int, [default=6]
        Depth of the trees in the models.
        range: [1,16]
    sigma : float, [default=0.1]
        Scale of GP kernel (lower values lead to lower posterior variance)
        range: (0,+inf)
    delta : float, [default=0]
        Scale of homogenious noise of GP kernel (adjust if target is noisy)
        range: [0,+inf)
    random_strength : float, [default=0.1]
        Corresponds to parameter beta in the paper. Higher values lead to faster convergence to GP posterior.
        range: (0,+inf)
    random_score_type : string [default='Gumbel']
        Type of random noise added to scores.
        Possible values:
            - 'Gumbel' - Gumbel-distributed (as in paper)
            - 'NormalWithModelSizeDecrease' - Normally-distributed with deviation decreasing with model iteration count (default in CatBoost)
    eps : float, [default=1e-4]
        Technical parameter that controls the precision of the prior estimation.
        range: (0, 1]
    verbose : bool or int
        Verbosity of posterior model training output
        If verbose is bool, then if set to True, logging_level is set to Verbose,
        if set to False, logging_level is set to Silent.
        If verbose is int, it determines the frequency of writing metrics to output and
        logging_level is set to Verbose.

    Returns
    -------
    models : list of trained CatBoostRegressor models (size = samples parameter value)
    """
    assert sigma > 0
    assert samples > 0
    assert random_strength > 0
    assert eps > 0
    if random_seed is None:
        random_seed = 0
    random_generator = np.random.default_rng(random_seed)
    prior_seeds = random_generator.integers(low=0, high=2 ** 63 - 1, size=samples)
    posterior_seeds = random_generator.integers(low=0, high=2 ** 63 - 1, size=samples)
    N = len(X)
    model_shrink_rate = (random_strength / sigma) ** 2 / N
    output_models = []
    tmp_file = tempfile.NamedTemporaryFile()
    prior_model_tmp_file = tmp_file.name
    for sample in range(samples):
        prior_y = random_generator.normal(scale=eps, size=N)
        prior = CatBoostRegressor(random_seed=prior_seeds[sample], iterations=prior_iterations, learning_rate=eps, loss_function='RMSE', bootstrap_type='No', depth=depth, verbose=False, leaf_estimation_backtracking='No', boost_from_average=False, random_strength=1 / eps, random_score_type=random_score_type, l2_leaf_reg=0, score_function='L2', boosting_type='Plain')
        prior.fit(X, prior_y, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, use_best_model=False)
        prior.save_model(prior_model_tmp_file, format='json', pool=X)
        with open(prior_model_tmp_file, 'r', encoding='utf-8') as prior_file:
            prior_json = json.load(prior_file)
        for tree in prior_json['oblivious_trees']:
            for ind, (val, weight) in enumerate(zip(tree['leaf_values'], tree['leaf_weights'])):
                tree['leaf_values'][ind] = random_generator.normal(scale=np.sqrt(N / np.sqrt(max(1, weight))))
        with open(prior_model_tmp_file, 'w') as prior_file:
            json.dump(prior_json, prior_file)
        prior.load_model(prior_model_tmp_file, format='json')
        scale, bias = prior.get_scale_and_bias()
        prior.set_scale_and_bias(scale * sigma / np.sqrt(prior_iterations), bias * sigma / np.sqrt(prior_iterations))
        posterior_y = y - prior.predict(X) + random_generator.normal(scale=delta, size=N)
        posterior = CatBoostRegressor(random_seed=posterior_seeds[sample], iterations=posterior_iterations, learning_rate=learning_rate, model_shrink_rate=model_shrink_rate, loss_function='RMSE', bootstrap_type='No', depth=depth, verbose=verbose, leaf_estimation_backtracking='No', boost_from_average=False, random_strength=random_strength, random_score_type=random_score_type, l2_leaf_reg=0, score_function='L2', boosting_type='Plain')
        posterior.fit(X, posterior_y, eval_set=eval_set, cat_features=cat_features, text_features=text_features, embedding_features=embedding_features, use_best_model=False)
        output_models.append(sum_models([prior, posterior], weights=[1, 1]))
    return output_models