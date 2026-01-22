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
def test_authentication_ignore_default_realm(self):
    """Verify that default realm is not used if realm present in
        username
        """
    addr = 'amqp://joe@myrealm:secret@%s:%d' % (self._broker.host, self._broker.port)
    self.config(sasl_default_realm='bad-realm', group='oslo_messaging_amqp')
    self._authentication_test(addr)