import os
import sys
import errno
import atexit
from warnings import warn
from looseversion import LooseVersion
import configparser
import numpy as np
from simplejson import load, dump
from .misc import str2bool
from filelock import SoftFileLock
def get_display(self):
    """Returns the first display available"""
    if self._display is not None:
        return ':%d' % self._display.new_display
    sysdisplay = None
    if self._config.has_option('execution', 'display_variable'):
        sysdisplay = self._config.get('execution', 'display_variable')
    sysdisplay = sysdisplay or os.getenv('DISPLAY')
    if sysdisplay:
        from collections import namedtuple

        def _mock():
            pass
        ndisp = sysdisplay.split(':')[-1].split('.')[0]
        Xvfb = namedtuple('Xvfb', ['new_display', 'stop'])
        self._display = Xvfb(int(ndisp), _mock)
        return self.get_display()
    else:
        if 'darwin' in sys.platform:
            raise RuntimeError('Xvfb requires root permissions to run in OSX. Please make sure that an X server is listening and set the appropriate config on either $DISPLAY or nipype\'s "display_variable" config. Valid X servers include VNC, XQuartz, or manually started Xvfb.')
        if sysdisplay == '':
            del os.environ['DISPLAY']
        try:
            from xvfbwrapper import Xvfb
        except ImportError:
            raise RuntimeError('A display server was required, but $DISPLAY is not defined and Xvfb could not be imported.')
        self._display = Xvfb(nolisten='tcp')
        self._display.start()
        if not hasattr(self._display, 'new_display'):
            setattr(self._display, 'new_display', self._display.vdisplay_num)
        return self.get_display()