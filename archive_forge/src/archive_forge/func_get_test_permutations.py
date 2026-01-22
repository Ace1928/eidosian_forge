import os
import random
import sys
import time
import xml.sax
import xml.sax.handler
from io import StringIO
from breezy import errors, osutils, trace, transport
from breezy.transport.http import urllib
def get_test_permutations():
    """Return the permutations to be used in testing."""
    from .tests import dav_server
    return [(HttpDavTransport, dav_server.DAVServer), (HttpDavTransport, dav_server.QuirkyDAVServer)]