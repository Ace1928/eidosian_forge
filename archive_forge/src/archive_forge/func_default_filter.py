import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def default_filter(name, obj):
    """Attribute filter.

    It filters out module attributes, and also methods starting with an
    underscore ``_``.

    This is used as the default filter for the :func:`create_patches` function
    and the :func:`patches` decorator.

    Parameters
    ----------
    name : str
        Attribute name.
    obj : object
        Attribute value.

    Returns
    -------
    bool
        Whether the attribute should be returned.
    """
    return not (isinstance(obj, types.ModuleType) or name.startswith('_'))