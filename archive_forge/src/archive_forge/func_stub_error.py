import threading
from unittest import mock
import eventlet
import fixtures
from oslo_config import cfg
from oslo_utils import eventletutils
import testscenarios
import oslo_messaging
from oslo_messaging import rpc
from oslo_messaging.rpc import dispatcher
from oslo_messaging.rpc import server as rpc_server_module
from oslo_messaging import server as server_module
from oslo_messaging.tests import utils as test_utils
def stub_error(msg, *a, **kw):
    if a and len(a) == 1 and isinstance(a[0], dict) and a[0]:
        a = a[0]
    errors.append(str(msg) % a)