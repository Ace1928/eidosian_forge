def check_traitsui_major_version(major):
    """ Raise RuntimeError if TraitsUI major version is less than the required
    value.

    Used internally in traits only.

    Parameters
    ----------
    major : int
        Required TraitsUI major version.

    Raises
    ------
    RuntimeError
    """
    from traitsui import __version__ as traitsui_version
    actual_major, _ = traitsui_version.split('.', 1)
    actual_major = int(actual_major)
    if actual_major < major:
        raise RuntimeError('TraitsUI {} or higher is required. Got version {!r}'.format(major, traitsui_version))