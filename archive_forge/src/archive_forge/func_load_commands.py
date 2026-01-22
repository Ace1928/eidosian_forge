import pkg_resources
import argparse
import logging
import sys
from warnings import warn
def load_commands(self):
    for ep in pkg_resources.iter_entry_points('pecan.command'):
        log.debug('%s loading plugin %s', self.__class__.__name__, ep)
        if ep.name in self.commands:
            warn('Duplicate entry points found on `%s` - ignoring %s' % (ep.name, ep), RuntimeWarning)
            continue
        try:
            cmd = ep.load()
            cmd.run
        except Exception as e:
            warn('Unable to load plugin %s: %s' % (ep, e), RuntimeWarning)
            continue
        self.add({ep.name: cmd})