import argparse
import os
from os import path as op
import shutil
import sys
from . import plugins
from .core import util
def remove_bin(plugin_names=['all']):
    """Remove binary dependencies of plugins

    This is a convenience method that removes all binaries
    dependencies for plugins downloaded by imageio.

    Notes
    -----
    It only makes sense to use this method if the binaries
    are corrupt.
    """
    if plugin_names.count('all'):
        plugin_names = PLUGINS_WITH_BINARIES
    print('Removing binaries for: {}.'.format(', '.join(plugin_names)))
    rdirs = util.resource_dirs()
    for plg in plugin_names:
        if plg not in PLUGINS_WITH_BINARIES:
            msg = 'Plugin {} not registered for binary download!'.format(plg)
            raise Exception(msg)
    not_removed = []
    for rd in rdirs:
        for rsub in os.listdir(rd):
            if rsub in plugin_names:
                plgdir = op.join(rd, rsub)
                try:
                    shutil.rmtree(plgdir)
                except Exception:
                    not_removed.append(plgdir)
    if not_removed:
        nrs = ','.join(not_removed)
        msg2 = 'These plugins files could not be removed: {}\n'.format(nrs) + 'Make sure they are not used by any program and try again.'
        raise Exception(msg2)