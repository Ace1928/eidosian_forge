import collections
import os
import tempfile
import time
import urllib
import uuid
import fixtures
from keystoneauth1 import loading as ks_loading
from oslo_config import cfg
from requests import structures
from requests_mock.contrib import fixture as rm_fixture
import openstack.cloud
import openstack.config as occ
import openstack.connection
from openstack.fixture import connection as os_fixture
from openstack.tests import base
from openstack.tests import fakes
def register_uris(self, uri_mock_list=None):
    """Mock a list of URIs and responses via requests mock.

        This method may be called only once per test-case to avoid odd
        and difficult to debug interactions. Discovery and Auth request mocking
        happens separately from this method.

        :param uri_mock_list: List of dictionaries that template out what is
                              passed to requests_mock fixture's `register_uri`.
                              Format is:
                                  {'method': <HTTP_METHOD>,
                                   'uri': <URI to be mocked>,
                                   ...
                                  }

                              Common keys to pass in the dictionary:
                                  * json: the json response (dict)
                                  * status_code: the HTTP status (int)
                                  * validate: The request body (dict) to
                                              validate with assert_calls
                              all key-word arguments that are valid to send to
                              requests_mock are supported.

                              This list should be in the order in which calls
                              are made. When `assert_calls` is executed, order
                              here will be validated. Duplicate URIs and
                              Methods are allowed and will be collapsed into a
                              single matcher. Each response will be returned
                              in order as the URI+Method is hit.
        :type uri_mock_list: list
        :return: None
        """
    assert not self.__register_uris_called
    self.__do_register_uris(uri_mock_list or [])
    self.__register_uris_called = True