import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_endpoints(attrs=None, count=2):
    """Create multiple fake endpoints.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of endpoints to fake
        :return:
            A list of FakeResource objects faking the endpoints
        """
    endpoints = []
    for i in range(0, count):
        endpoints.append(FakeEndpoint.create_one_endpoint(attrs))
    return endpoints