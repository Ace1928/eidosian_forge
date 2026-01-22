import errno
import io
from unittest import mock
import sys
import uuid
import fixtures
from oslo_config import cfg
from oslo_utils import units
import glance_store as store
from glance_store import exceptions
from glance_store import location
from glance_store.tests import base
from glance_store.tests.unit.cinder import test_cinder_base
from glance_store.tests.unit import test_store_capabilities as test_cap
from glance_store._drivers.cinder import store as cinder # noqa
def fake_volume_type_check(name):
    raise cinder.exceptions.AuthorizationFailure(code=401)