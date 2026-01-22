import datetime
from unittest import mock
from oslo_utils import timeutils
from keystoneclient import access
from keystoneclient import httpclient
from keystoneclient.tests.unit import utils
from keystoneclient.tests.unit.v2_0 import client_fixtures
from keystoneclient import utils as client_utils
A Simple testing keyring.

            This class supports stubbing an initial password to be returned by
            setting password, and allows easy password and key retrieval. Also
            records if a password was retrieved.
            