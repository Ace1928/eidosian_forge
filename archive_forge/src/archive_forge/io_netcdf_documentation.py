from .converters import convert_to_inference_data
from .inference_data import InferenceData
Save dataset as a netcdf file.

    WARNING: Only idempotent in case `data` is InferenceData

    Parameters
    ----------
    data : InferenceData, or any object accepted by `convert_to_inference_data`
        Object to be saved
    filename : str
        name or path of the file to load trace
    group : str (optional)
        In case `data` is not InferenceData, this is the group it will be saved to
    engine : {"h5netcdf", "netcdf4"}, default "h5netcdf"
        Library used to read the netcdf file.
    coords : dict (optional)
        See `convert_to_inference_data`
    dims : dict (optional)
        See `convert_to_inference_data`

    Returns
    -------
    str
        filename saved to
    