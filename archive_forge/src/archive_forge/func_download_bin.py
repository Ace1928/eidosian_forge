import argparse
import os
from os import path as op
import shutil
import sys
from . import plugins
from .core import util
def download_bin(plugin_names=['all'], package_dir=False):
    """Download binary dependencies of plugins

    This is a convenience method for downloading the binaries
    (e.g. for freeimage) from the imageio-binaries
    repository.

    Parameters
    ----------
    plugin_names: list
        A list of imageio plugin names. If it contains "all", all
        binary dependencies are downloaded.
    package_dir: bool
        If set to `True`, the binaries will be downloaded to the
        `resources` directory of the imageio package instead of
        to the users application data directory. Note that this
        might require administrative rights if imageio is installed
        in a system directory.
    """
    if plugin_names.count('all'):
        plugin_names = PLUGINS_WITH_BINARIES
    plugin_names.sort()
    print('Ascertaining binaries for: {}.'.format(', '.join(plugin_names)))
    if package_dir:
        directory = util.resource_package_dir()
    else:
        directory = None
    for plg in plugin_names:
        if plg not in PLUGINS_WITH_BINARIES:
            msg = 'Plugin {} not registered for binary download!'.format(plg)
            raise Exception(msg)
        mod = getattr(plugins, plg)
        mod.download(directory=directory)