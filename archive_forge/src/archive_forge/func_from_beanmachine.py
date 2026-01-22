from .inference_data import InferenceData
from .base import dict_to_dataset, requires
def from_beanmachine(sampler=None, *, coords=None, dims=None):
    """Convert Bean Machine MonteCarloSamples object into an InferenceData object.

    For a usage example read the
    :ref:`Creating InferenceData section on from_beanmachine <creating_InferenceData>`


    Parameters
    ----------
    sampler : bm.MonteCarloSamples
        Fitted MonteCarloSamples object from Bean Machine
    coords : dict of {str : array-like}
        Map of dimensions to coordinates
    dims : dict of {str : list of str}
        Map variable names to their coordinates

    Warnings
    --------
    `beanmachine` is no longer under active development, and therefore, it
    is not possible to test this converter in ArviZ's CI.
    """
    return BMConverter(sampler=sampler, coords=coords, dims=dims).to_inference_data()