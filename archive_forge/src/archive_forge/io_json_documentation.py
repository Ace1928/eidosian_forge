from .io_dict import from_dict
Save dataset as a json file.

    Will use the faster `ujson` (https://github.com/ultrajson/ultrajson) if it is available.

    WARNING: Only idempotent in case `idata` is InferenceData.

    Parameters
    ----------
    idata : InferenceData
        Object to be saved
    filename : str
        name or path of the file to load trace

    Returns
    -------
    str
        filename saved to
    