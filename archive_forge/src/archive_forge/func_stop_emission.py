import functools
import warnings
from collections import namedtuple
import gi.module
from gi.overrides import override, deprecated_attr
from gi.repository import GLib
from gi import PyGIDeprecationWarning
from gi import _propertyhelper as propertyhelper
from gi import _signalhelper as signalhelper
from gi import _gi
from gi import _option as option
def stop_emission(self, detailed_signal):
    """Deprecated, please use stop_emission_by_name."""
    warnings.warn(self.stop_emission.__doc__, PyGIDeprecationWarning, stacklevel=2)
    return self.stop_emission_by_name(detailed_signal)