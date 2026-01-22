from .converters import convert_to_inference_data
from .inference_data import InferenceData
def from_netcdf(filename, *, engine='h5netcdf', group_kwargs=None, regex=False):
    """Load netcdf file back into an arviz.InferenceData.

    Parameters
    ----------
    filename : str
        name or path of the file to load trace
    engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
        Library used to read the netcdf file.
    group_kwargs : dict of {str: dict}
        Keyword arguments to be passed into each call of :func:`xarray.open_dataset`.
        The keys of the higher level should be group names or regex matching group
        names, the inner dicts re passed to ``open_dataset``.
        This feature is currently experimental
    regex : str
        Specifies where regex search should be used to extend the keyword arguments.

    Returns
    -------
        InferenceData object

    Notes
    -----
    By default, the datasets of the InferenceData object will be lazily loaded instead
    of loaded into memory. This behaviour is regulated by the value of
    ``az.rcParams["data.load"]``.
    """
    if group_kwargs is None:
        group_kwargs = {}
    return InferenceData.from_netcdf(filename, engine=engine, group_kwargs=group_kwargs, regex=regex)