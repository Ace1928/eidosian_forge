import math
from collections import OrderedDict
import numpy
from .base import numeric_types, string_types
from . import ndarray
from . import registry
def check_label_shapes(labels, preds, wrap=False, shape=False):
    """Helper function for checking shape of label and prediction

    Parameters
    ----------
    labels : list of `NDArray`
        The labels of the data.

    preds : list of `NDArray`
        Predicted values.

    wrap : boolean
        If True, wrap labels/preds in a list if they are single NDArray

    shape : boolean
        If True, check the shape of labels and preds;
        Otherwise only check their length.
    """
    if not shape:
        label_shape, pred_shape = (len(labels), len(preds))
    else:
        label_shape, pred_shape = (labels.shape, preds.shape)
    if label_shape != pred_shape:
        raise ValueError('Shape of labels {} does not match shape of predictions {}'.format(label_shape, pred_shape))
    if wrap:
        if isinstance(labels, ndarray.ndarray.NDArray):
            labels = [labels]
        if isinstance(preds, ndarray.ndarray.NDArray):
            preds = [preds]
    return (labels, preds)