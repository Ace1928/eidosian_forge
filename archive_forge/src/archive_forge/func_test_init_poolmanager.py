import argparse
from io import StringIO
import itertools
import logging
from unittest import mock
import uuid
from oslo_config import cfg
from oslo_config import fixture as config
from oslo_serialization import jsonutils
import requests
from testtools import matchers
from keystoneclient import adapter
from keystoneclient.auth import base
from keystoneclient import exceptions
from keystoneclient.i18n import _
from keystoneclient import session as client_session
from keystoneclient.tests.unit import utils
@mock.patch.object(client_session, 'socket')
@mock.patch('requests.adapters.HTTPAdapter.init_poolmanager')
def test_init_poolmanager(self, mock_parent_init_poolmanager, mock_socket):
    spec = ['IPPROTO_TCP', 'TCP_NODELAY', 'SOL_SOCKET', 'SO_KEEPALIVE']
    mock_socket.mock_add_spec(spec)
    adapter = client_session.TCPKeepAliveAdapter()
    adapter.init_poolmanager()
    call_args, call_kwargs = mock_parent_init_poolmanager.call_args
    called_socket_opts = call_kwargs['socket_options']
    call_options = [opt for protocol, opt, value in called_socket_opts]
    self.assertEqual([mock_socket.TCP_NODELAY, mock_socket.SO_KEEPALIVE], call_options)