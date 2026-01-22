import atexit
import codecs
import contextlib
import copy
import difflib
import doctest
import errno
import functools
import itertools
import logging
import math
import os
import platform
import pprint
import random
import re
import shlex
import site
import stat
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import unittest
import warnings
from io import BytesIO, StringIO, TextIOWrapper
from typing import Callable, Set
import testtools
from testtools import content
import breezy
from breezy.bzr import chk_map
from .. import branchbuilder
from .. import commands as _mod_commands
from .. import config, controldir, debug, errors, hooks, i18n
from .. import lock as _mod_lock
from .. import lockdir, osutils
from .. import plugin as _mod_plugin
from .. import pyutils, registry, symbol_versioning, trace
from .. import transport as _mod_transport
from .. import ui, urlutils, workingtree
from ..bzr.smart import client, request
from ..tests import TestUtil, fixtures, test_server, treeshape, ui_testing
from ..transport import memory, pathfilter
from testtools.testcase import TestSkipped
def setup_smart_server_with_call_log(self):
    """Sets up a smart server as the transport server with a call log."""
    self.transport_server = test_server.SmartTCPServer_for_testing
    self.hpss_connections = []
    self.hpss_calls = []
    import traceback
    prefix_length = len(traceback.extract_stack()) - 2

    def capture_hpss_call(params):
        self.hpss_calls.append(CapturedCall(params, prefix_length))

    def capture_connect(transport):
        self.hpss_connections.append(transport)
    client._SmartClient.hooks.install_named_hook('call', capture_hpss_call, None)
    _mod_transport.Transport.hooks.install_named_hook('post_connect', capture_connect, None)