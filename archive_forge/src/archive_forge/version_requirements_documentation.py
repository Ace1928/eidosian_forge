import sys
from packaging import version as _version
Return a module object of name *module_name* if installed.

    Parameters
    ----------
    module_name : str
        Name of module.
    version : str, optional
        Version string to test against.
        If version is not None, checking version
        (must have an attribute named '__version__' or 'VERSION')
        Version may start with =, >=, > or < to specify the exact requirement

    Returns
    -------
    mod : module or None
        Module if *module_name* is installed matching the optional version
        or None otherwise.
    