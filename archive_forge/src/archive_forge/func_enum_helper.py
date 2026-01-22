import os
import platform
import sys
from functools import partial
import zmq
from packaging.version import Version as V
from traitlets.config.application import Application
@lru_cache(None)
def enum_helper(name):
    return operator.attrgetter(name.rpartition('.')[0])(sys.modules[QtCore.__package__])