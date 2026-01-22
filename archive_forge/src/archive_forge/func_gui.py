from logging import error
import io
import os
from pprint import pformat
import sys
from warnings import warn
from traitlets.utils.importstring import import_item
from IPython.core import magic_arguments, page
from IPython.core.error import UsageError
from IPython.core.magic import Magics, magics_class, line_magic, magic_escapes
from IPython.utils.text import format_screen, dedent, indent
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.ipstruct import Struct
@line_magic
def gui(self, parameter_s=''):
    """Enable or disable IPython GUI event loop integration.

        %gui [GUINAME]

        This magic replaces IPython's threaded shells that were activated
        using the (pylab/wthread/etc.) command line flags.  GUI toolkits
        can now be enabled at runtime and keyboard
        interrupts should work without any problems.  The following toolkits
        are supported:  wxPython, PyQt4, PyGTK, Tk and Cocoa (OSX)::

            %gui wx      # enable wxPython event loop integration
            %gui qt      # enable PyQt/PySide event loop integration
                         # with the latest version available.
            %gui qt6     # enable PyQt6/PySide6 event loop integration
            %gui qt5     # enable PyQt5/PySide2 event loop integration
            %gui gtk     # enable PyGTK event loop integration
            %gui gtk3    # enable Gtk3 event loop integration
            %gui gtk4    # enable Gtk4 event loop integration
            %gui tk      # enable Tk event loop integration
            %gui osx     # enable Cocoa event loop integration
                         # (requires %matplotlib 1.1)
            %gui         # disable all event loop integration

        WARNING:  after any of these has been called you can simply create
        an application object, but DO NOT start the event loop yourself, as
        we have already handled that.
        """
    opts, arg = self.parse_options(parameter_s, '')
    if arg == '':
        arg = None
    try:
        return self.shell.enable_gui(arg)
    except Exception as e:
        error(str(e))