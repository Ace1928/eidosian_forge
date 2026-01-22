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
def filter_same_topic_and_server(scenario):
    params = scenario[1]
    single_topic = params['topic1'] == params['topic2']
    single_server = params['server1'] == params['server2']
    return not (single_topic and single_server)