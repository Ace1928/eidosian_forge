import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def plugin_order():
    """Return the currently preferred plugin order.

    Returns
    -------
    p : dict
        Dictionary of preferred plugin order, with function name as key and
        plugins (in order of preference) as value.

    """
    p = {}
    for func in plugin_store:
        p[func] = [plugin_name for plugin_name, f in plugin_store[func]]
    return p