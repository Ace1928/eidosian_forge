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
def plot_predictions(self, data, features_to_change, plot=True, plot_file=None):
    """
        To use this function, you should install plotly.

        data: numpy.ndarray or pandas.DataFrame or catboost.Pool
        features_to_change: list-like with int (for indices) or str (for names) elements
            Numerical features indices or names in `data` for which you want to vary prediction value.
        plot: bool
            Plot predictions.
        plot_file: str
            Output file for plot predictions.
        Returns
        -------
            List of list of predictions for all buckets for all samples in data
        """

    def predict(doc, feature_idx, borders, nan_treatment):
        left_extend_border = min(2 * borders[0], -1)
        right_extend_border = max(2 * borders[-1], 1)
        extended_borders = [left_extend_border] + borders + [right_extend_border]
        points = []
        predictions = []
        border_idx = None
        if np.isnan(doc[feature_idx]):
            border_idx = len(borders) if nan_treatment == 'AsTrue' else 0
        for i in range(len(extended_borders) - 1):
            points += [(extended_borders[i] + extended_borders[i + 1]) / 2.0]
            if border_idx is None and doc[feature_idx] < extended_borders[i + 1]:
                border_idx = i
            buf = doc[feature_idx]
            doc[feature_idx] = points[-1]
            predictions += [self.predict(doc)]
            doc[feature_idx] = buf
        if border_idx is None:
            border_idx = len(borders)
        return (predictions, border_idx)

    def get_layout(go, feature, xaxis):
        return go.Layout(title="Prediction variation for feature '{}'".format(feature), yaxis={'title': 'Prediction', 'side': 'left', 'overlaying': 'y2'}, xaxis=xaxis)
    try:
        import plotly.graph_objs as go
    except ImportError as e:
        warnings.warn('To draw plots you should install plotly.')
        raise ImportError(str(e))
    model_borders = self._get_borders()
    data, _ = self._process_predict_input_data(data, 'vary_feature_value_and_apply', thread_count=-1)
    figs = []
    all_predictions = [{}] * data.num_row()
    nan_treatments = self._get_nan_treatments()
    for feature in features_to_change:
        if not isinstance(feature, int):
            if self.feature_names_ is None or feature not in self.feature_names_:
                raise CatBoostError('No feature named "{}" in model'.format(feature))
            feature_idx = self.feature_names_.index(feature)
        else:
            feature_idx = feature
            feature = self.feature_names_[feature_idx]
        assert feature_idx in model_borders, 'only float features indexes are supported'
        borders = model_borders[feature_idx]
        if len(borders) == 0:
            xaxis = go.layout.XAxis(title='Bins', tickvals=[0])
            figs += go.Figure(data=[], layout=get_layout(go, feature_idx, xaxis))
        xaxis = go.layout.XAxis(title='Bins', tickmode='array', tickvals=list(range(len(borders) + 1)), ticktext=['(-inf, {:.4f}]'.format(borders[0])] + ['({:.4f}, {:.4f}]'.format(val_1, val_2) for val_1, val_2 in zip(borders[:-1], borders[1:])] + ['({:.4f}, +inf)'.format(borders[-1])], showticklabels=False)
        trace = []
        for idx, features in enumerate(data.get_features()):
            predictions, border_idx = predict(features, feature_idx, borders, nan_treatments[feature_idx])
            all_predictions[idx][feature_idx] = predictions
            trace.append(go.Scatter(y=predictions, mode='lines+markers', name=u'Document {} predictions'.format(idx)))
            trace.append(go.Scatter(x=[border_idx], y=[predictions[border_idx]], showlegend=False))
        layout = get_layout(go, feature, xaxis)
        figs += [go.Figure(data=trace, layout=layout)]
    if plot:
        try_plot_offline(figs)
    if plot_file:
        save_plot_file(plot_file, 'Predictions for all buckets', figs)
    return (all_predictions, figs)