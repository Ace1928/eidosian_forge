import numpy as np
from ..processors import SequentialProcessor
def key_prediction_to_label(prediction):
    """
    Convert key class id to a human-readable key name.

    Parameters
    ----------
    prediction : numpy array
        Array containing the probabilities of each key class.

    Returns
    -------
    str
        Human-readable key name.

    """
    prediction = np.atleast_2d(prediction)
    return KEY_LABELS[prediction[0].argmax()]