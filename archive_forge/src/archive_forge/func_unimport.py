import os
import sys
import pytest
import zmq
def unimport():
    pyximport.uninstall(*importers)
    sys.modules.pop('zmq.tests.cython_ext', None)