import contextlib
import io
import socket
from unittest import mock
import netaddr
import netifaces
from oslotest import base as test_base
from oslo_utils import netutils
def reset_detection_flag():
    netutils._IS_IPV6_ENABLED = None