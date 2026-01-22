import copy
import logging
import os
import queue
import select
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest import mock
import uuid
from oslo_utils import eventletutils
from oslo_utils import importutils
from string import Template
import testtools
import oslo_messaging
from oslo_messaging.tests import utils as test_utils
def test_broker_shutdown(self):
    """Simulate a normal shutdown of a broker."""

    def _meth(broker):
        broker.stop(clean=True)
        time.sleep(0.5)
    self._failover(_meth)