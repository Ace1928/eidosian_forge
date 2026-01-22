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
def calc_feature_statistics(self, data, target=None, feature=None, prediction_type=None, cat_feature_values=None, plot=True, max_cat_features_on_plot=10, thread_count=-1, plot_file=None):
    """
        Get statistics for the feature using the model, dataset and target.
        To use this function, you should install plotly.

        The catboost model has borders for the float features used in it. The borders divide
        feature values into bins, and the model's prediction depends on the number of the bin where the
        feature value falls in.

        For float features this function takes model's borders and computes
        1) Mean target value for every bin;
        2) Mean model prediction for every bin;
        3) The number of objects in dataset which fall into each bin;
        4) Predictions on varying feature. For every object, varies the feature value
        so that it falls into bin #0, bin #1, ... and counts model predictions.
        Then counts average prediction for each bin.

        For categorical features (only one-hot supported) does the same, but takes feature values
        provided in cat_feature_values instead of borders.

        Parameters
        ----------
        data: numpy.ndarray or pandas.DataFrame or catboost. Pool or dict {'pool_name': pool} if you want several pools
            Data to compute statistics on
        target: numpy.ndarray or pandas.Series or dict {'pool_name': target} if you want several pools or None
            Target corresponding to data
            Use only if data is not catboost.Pool.
        feature: None, int, string, or list of int or strings
            Features indexes or names in pd.DataFrame for which you want to get statistics.
            None, if you need statistics for all features.
        prediction_type: str
            Prediction type used for counting mean_prediction: 'Class', 'Probability' or 'RawFormulaVal'.
            If not specified, is derived from the model.
        cat_feature_values: list or numpy.ndarray or pandas.Series or
                            dict: int or string to list or numpy.ndarray or pandas.Series
            Contains categorical feature values you need to get statistics on.
            Use dict, when parameter 'feature' is a list to specify cat values for different features.
            When parameter 'feature' is int or str, you can just pass list of cat values.
        plot: bool
            Plot statistics.
        max_cat_features_on_plot: int
            If categorical feature takes more than max_cat_features_on_plot different unique values,
            output result on several plots, not more than max_cat_features_on_plot feature values on each.
            Used only if plot=True or plot_file is not None.
        thread_count: int
            Number of threads to use for getting statistics.
        plot_file: str
            Output file for plot statistics.

        Returns
        -------
        dict if parameter 'feature' is int or string, else dict of dicts:
            For each unique feature contain
            python dict with binarized feature statistics.
            For float feature, includes
                    'borders' -- borders for the specified feature in model
                    'binarized_feature' -- numbers of bins where feature values fall
                    'mean_target' -- mean value of target over each bin
                    'mean_prediction' -- mean value of model prediction over each bin
                    'objects_per_bin' -- number of objects per bin
                    'predictions_on_varying_feature' -- averaged over dataset predictions for
                    varying feature (see above)
            For one-hot feature, returns the same, but with 'cat_values' instead of 'borders'
        """
    target_is_none = target is None
    if not isinstance(data, dict):
        data = {'': data}
    if not isinstance(target, dict):
        target = {'': target}
    assert target_is_none or len(data) == len(target), 'inconsistent size of data and target'
    assert target_is_none or target.keys() == data.keys(), 'inconsistent pool_names of data and target'
    for key in data.keys():
        data[key], _ = self._process_predict_input_data(data[key], 'get_binarized_statistics', thread_count, target.get(key, None))
    data, pool_names = (list(data.values()), list(data.keys()))
    if prediction_type is None:
        prediction_type = 'Probability' if self.get_param('loss_function') in ['CrossEntropy', 'Logloss'] else 'RawFormulaVal'
    if prediction_type not in ['Class', 'Probability', 'RawFormulaVal', 'Exponent']:
        raise CatBoostError('Unknown prediction type "{}"'.format(prediction_type))
    if feature is None:
        feature = self.feature_names_
    if cat_feature_values is None:
        cat_feature_values = {}
    elif not isinstance(cat_feature_values, dict):
        if isinstance(feature, list):
            raise CatBoostError('cat_feature_values should be dict when features is a list')
        else:
            cat_feature_values = {feature: cat_feature_values}
    if isinstance(feature, str) or isinstance(feature, int):
        features = [feature]
        is_for_one_feature = True
    else:
        features = feature
        is_for_one_feature = False
    cat_features_nums = []
    float_features_nums = []
    feature_type_mapper = []
    feature_names = []
    feature_name_to_num = {}
    for feature in features:
        if not isinstance(feature, int):
            if self.feature_names_ is None or feature not in self.feature_names_:
                raise CatBoostError('No feature named "{}" in model'.format(feature))
            feature_num = self.feature_names_.index(feature)
        else:
            feature_num = feature
            feature = self.feature_names_[feature_num]
        if feature in feature_names:
            continue
        if feature_num in cat_feature_values:
            cat_feature_values[feature] = cat_feature_values[feature_num]
        feature_names.append(feature)
        feature_name_to_num[feature] = feature_num
        feature_type, feature_internal_index = self._object._get_feature_type_and_internal_index(feature_num)
        if feature_type == 'categorical':
            cat_features_nums.append(feature_internal_index)
            feature_type_mapper.append('cat')
        else:
            float_features_nums.append(feature_internal_index)
            feature_type_mapper.append('float')
    results = [self._object._get_binarized_statistics(data_item, cat_features_nums, float_features_nums, prediction_type, thread_count) for data_item in data]
    statistics_by_feature = defaultdict(list)
    to_float_offset = len(cat_features_nums)
    cat_index, float_index = (0, to_float_offset)
    for i, type in enumerate(feature_type_mapper):
        feature_name = feature_names[i]
        feature_num = feature_name_to_num[feature_name]
        if type == 'cat':
            if feature_name not in cat_feature_values:
                cat_feature_values_ = self._object._get_cat_feature_values(data[0], feature_num)
                cat_feature_values_ = [val for val in cat_feature_values_]
            else:
                cat_feature_values_ = cat_feature_values[feature_name]
            if not isinstance(cat_feature_values_, ARRAY_TYPES):
                raise CatBoostError("Feature '{}' is categorical. Please provide values for which you need statistics in cat_feature_values".format(feature))
            val_to_hash = dict()
            for val in cat_feature_values_:
                val_to_hash[val] = self._object._calc_cat_feature_perfect_hash(val, cat_features_nums[cat_index])
            hash_to_val = {hash: val for val, hash in val_to_hash.items()}
            for i, res in enumerate(results):
                res[cat_index]['cat_values'] = np.array([hash_to_val[i] for i in sorted(hash_to_val.keys())])
                res[cat_index].pop('borders', None)
                statistics_by_feature[feature_num].append(res[cat_index])
            cat_index += 1
        else:
            for res in results:
                statistics_by_feature[feature_num].append(res[float_index])
            float_index += 1
    if plot or plot_file is not None:
        fig = _plot_feature_statistics(statistics_by_feature, pool_names, self.feature_names_, max_cat_features_on_plot)
        if plot:
            try_plot_offline(fig)
        if plot_file is not None:
            save_plot_file(plot_file, 'Catboost metrics graph', [fig])
    for key in statistics_by_feature.keys():
        if len(statistics_by_feature[key]) == 1:
            statistics_by_feature[key] = statistics_by_feature[key][0]
    if is_for_one_feature:
        return statistics_by_feature[feature_name_to_num[feature_names[0]]]
    return_stats = {}
    for feature in features:
        if isinstance(feature, int):
            return_stats[feature] = statistics_by_feature[feature]
        else:
            return_stats[feature] = statistics_by_feature[feature_name_to_num[feature]]
    return return_stats