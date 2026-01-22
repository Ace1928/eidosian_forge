from __future__ import absolute_import, division, print_function
import numpy as np
from . import layers, activations
from ...processors import Processor, ParallelProcessor, SequentialProcessor
def average_predictions(predictions):
    """
    Returns the average of all predictions.

    Parameters
    ----------
    predictions : list
        Predictions (i.e. NN activation functions).

    Returns
    -------
    numpy array
        Averaged prediction.

    """
    if len(predictions) > 1:
        predictions = sum(predictions) / len(predictions)
    else:
        predictions = predictions[0]
    return predictions