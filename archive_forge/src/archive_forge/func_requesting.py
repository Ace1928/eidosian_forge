import os
import platform
import socket
import tempfile
import testtools
from unittest import mock
import eventlet
import eventlet.wsgi
import requests
import webob
from oslo_config import cfg
from oslo_service import sslutils
from oslo_service.tests import base
from oslo_service import wsgi
from oslo_utils import netutils
def requesting(host, port, ca_certs=None, method='POST', content_type='application/x-www-form-urlencoded', address_familly=socket.AF_INET):
    frame = bytes('{verb} / HTTP/1.1\r\n\r\n'.format(verb=method), 'utf-8')
    with socket.socket(address_familly, socket.SOCK_STREAM) as sock:
        if ca_certs:
            with eventlet.wrap_ssl(sock, ca_certs=ca_certs) as wrappedSocket:
                wrappedSocket.connect((host, port))
                wrappedSocket.send(frame)
                data = wrappedSocket.recv(1024).decode()
                return data
        else:
            sock.connect((host, port))
            sock.send(frame)
            data = sock.recv(1024).decode()
            return data