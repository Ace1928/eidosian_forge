import glob
from itertools import chain
import os
import sys
from traitlets.config.application import boolean_flag
from traitlets.config.configurable import Configurable
from traitlets.config.loader import Config
from IPython.core.application import SYSTEM_CONFIG_DIRS, ENV_CONFIG_DIRS
from IPython.core import pylabtools
from IPython.utils.contexts import preserve_keys
from IPython.utils.path import filefind
from traitlets import (
from IPython.terminal import pt_inputhooks
def init_gui_pylab(self):
    """Enable GUI event loop integration, taking pylab into account."""
    enable = False
    shell = self.shell
    if self.pylab:
        enable = lambda key: shell.enable_pylab(key, import_all=self.pylab_import_all)
        key = self.pylab
    elif self.matplotlib:
        enable = shell.enable_matplotlib
        key = self.matplotlib
    elif self.gui:
        enable = shell.enable_gui
        key = self.gui
    if not enable:
        return
    try:
        r = enable(key)
    except ImportError:
        self.log.warning('Eventloop or matplotlib integration failed. Is matplotlib installed?')
        self.shell.showtraceback()
        return
    except Exception:
        self.log.warning('GUI event loop or pylab initialization failed')
        self.shell.showtraceback()
        return
    if isinstance(r, tuple):
        gui, backend = r[:2]
        self.log.info('Enabling GUI event loop integration, eventloop=%s, matplotlib=%s', gui, backend)
        if key == 'auto':
            print('Using matplotlib backend: %s' % backend)
    else:
        gui = r
        self.log.info('Enabling GUI event loop integration, eventloop=%s', gui)