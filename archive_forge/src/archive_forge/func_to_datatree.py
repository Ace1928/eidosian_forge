from .inference_data import InferenceData
def to_datatree(data):
    """Convert InferenceData object to a :class:`~datatree.DataTree`.

    Parameters
    ----------
    data : InferenceData
    """
    return data.to_datatree()