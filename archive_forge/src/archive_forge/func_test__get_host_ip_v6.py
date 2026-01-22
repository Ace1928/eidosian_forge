import contextlib
import hashlib
import io
import math
import os
from unittest import mock
import socket
import sys
import tempfile
import time
import uuid
from keystoneauth1 import exceptions as keystone_exc
from os_brick.initiator import connector
from oslo_concurrency import processutils
from oslo_utils.secretutils import md5
from oslo_utils import units
from glance_store._drivers.cinder import scaleio
from glance_store.common import attachment_state_manager
from glance_store.common import cinder_utils
from glance_store import exceptions
from glance_store import location
from glance_store._drivers.cinder import store as cinder # noqa
from glance_store._drivers.cinder import nfs # noqa
def test__get_host_ip_v6(self):
    fake_ipv6 = '2001:0db8:85a3:0000:0000:8a2e:0370'
    fake_socket_return = [[0, 1, 2, 3, [fake_ipv6]]]
    with mock.patch.object(cinder.socket, 'getaddrinfo') as fake_socket:
        fake_socket.return_value = fake_socket_return
        res = self.store._get_host_ip('fake_host')
        self.assertEqual(fake_ipv6, res)