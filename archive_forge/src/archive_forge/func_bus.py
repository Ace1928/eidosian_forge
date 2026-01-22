import os
import sys
import threading
import time
import unittest.mock
import pytest
from cherrypy.process import wspbus
@pytest.fixture
def bus():
    """Return a wspbus instance."""
    return wspbus.Bus()