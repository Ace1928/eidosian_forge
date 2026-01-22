import os.path
import warnings
from configparser import ConfigParser
from glob import glob
from .collection import imread_collection_wrapper
def find_available_plugins(loaded=False):
    """List available plugins.

    Parameters
    ----------
    loaded : bool
        If True, show only those plugins currently loaded.  By default,
        all plugins are shown.

    Returns
    -------
    p : dict
        Dictionary with plugin names as keys and exposed functions as
        values.

    """
    active_plugins = set()
    for plugin_func in plugin_store.values():
        for plugin, func in plugin_func:
            active_plugins.add(plugin)
    d = {}
    for plugin in plugin_provides:
        if not loaded or plugin in active_plugins:
            d[plugin] = [f for f in plugin_provides[plugin] if not f.startswith('_')]
    return d