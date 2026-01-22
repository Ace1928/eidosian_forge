import inspect
import numpy as np
import six
from keras_tuner.src import protos
from keras_tuner.src.api_export import keras_tuner_export
from keras_tuner.src.backend import keras
@keras_tuner_export('keras_tuner.engine.metrics_tracking.infer_metric_direction')
def infer_metric_direction(metric):
    if isinstance(metric, six.string_types):
        metric_name = metric
        if metric_name.startswith('val_'):
            metric_name = metric_name.replace('val_', '', 1)
        if metric_name.startswith('weighted_'):
            metric_name = metric_name.replace('weighted_', '', 1)
        if metric_name in {'loss', 'crossentropy', 'ce'}:
            return 'min'
        elif metric_name == 'acc':
            return 'max'
        try:
            if 'use_legacy_format' in inspect.getfullargspec(keras.metrics.deserialize).args:
                metric = keras.metrics.deserialize(metric_name, use_legacy_format=True)
            else:
                metric = keras.metrics.deserialize(metric_name)
        except ValueError:
            try:
                if 'use_legacy_format' in inspect.getfullargspec(keras.losses.deserialize).args:
                    metric = keras.losses.deserialize(metric_name, use_legacy_format=True)
                else:
                    metric = keras.losses.deserialize(metric_name)
            except Exception:
                return None
    if isinstance(metric, (keras.metrics.Metric, keras.losses.Loss)):
        name = metric.__class__.__name__
        if name == 'MeanMetricWrapper':
            name = metric._fn.__name__
    elif isinstance(metric, str):
        name = metric
    else:
        name = metric.__name__
    if name in _MAX_METRICS or name in _MAX_METRIC_FNS:
        return 'max'
    elif hasattr(keras.metrics, name) or hasattr(keras.losses, name):
        return 'min'
    return None