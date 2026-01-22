import socket
from unittest import mock
import uuid
from cinderclient.v3 import client as cinderclient
import glance_store
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import strutils
from glance.common import wsgi
from glance.tests import functional
Test to check if an image is successfully migrated when we upgrade
        from a single cinder store to multiple cinder stores, and that
        GETs from non-owners in the meantime are not interrupted.
        