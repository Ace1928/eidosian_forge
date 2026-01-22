import numpy as np
from sklearn.base import BaseEstimator, is_classifier, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect
def set_sample_weight(pipeline_steps, sample_weight=None):
    """Recursively iterates through all objects in the pipeline and sets sample weight.

    Parameters
    ----------
    pipeline_steps: array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
    sample_weight: array-like
        List of sample weight
    Returns
    -------
    sample_weight_dict:
        A dictionary of sample_weight

    """
    sample_weight_dict = {}
    if not isinstance(sample_weight, type(None)):
        for pname, obj in pipeline_steps:
            if inspect.getargspec(obj.fit).args.count('sample_weight'):
                step_sw = pname + '__sample_weight'
                sample_weight_dict[step_sw] = sample_weight
    if sample_weight_dict:
        return sample_weight_dict
    else:
        return None