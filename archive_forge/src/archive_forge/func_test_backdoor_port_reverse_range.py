import errno
import os
import socket
from unittest import mock
import eventlet
from oslo_service import eventlet_backdoor
from oslo_service.tests import base
def test_backdoor_port_reverse_range(self):
    self.config(backdoor_port='8888:7777')
    self.assertRaises(eventlet_backdoor.EventletBackdoorConfigValueError, eventlet_backdoor.initialize_if_enabled, self.conf)