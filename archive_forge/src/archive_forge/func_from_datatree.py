from .inference_data import InferenceData
def from_datatree(datatree):
    """Create an InferenceData object from a :class:`~datatree.DataTree`.

    Parameters
    ----------
    datatree : DataTree
    """
    return InferenceData.from_datatree(datatree)