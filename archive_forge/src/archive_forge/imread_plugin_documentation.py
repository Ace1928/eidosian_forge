from ...util.dtype import _convert
Save an image to disk.

    Parameters
    ----------
    fname : str
        Name of destination file.
    arr : ndarray of uint8 or uint16
        Array (image) to save.
    format_str : str,optional
        Format to save as.

    Notes
    -----
    Currently, only 8-bit precision is supported.
    