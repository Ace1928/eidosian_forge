import asyncio
import logging
import os
import shutil
import sys
import warnings
from contextlib import contextmanager
import pytest
import zmq
import zmq.asyncio
import zmq.auth
from zmq.tests import SkipTest, skip_pypy
@contextmanager
def curve_push_pull(self, certs, client_key='ok'):
    server_public, server_secret, client_public, client_secret = certs
    with self.push_pull() as (server, client):
        server.curve_publickey = server_public
        server.curve_secretkey = server_secret
        server.curve_server = True
        if client_key is not None:
            client.curve_publickey = client_public
            client.curve_secretkey = client_secret
            if client_key == 'ok':
                client.curve_serverkey = server_public
            else:
                private, public = zmq.curve_keypair()
                client.curve_serverkey = public
        yield (server, client)