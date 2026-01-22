import copy
from unittest import mock
import uuid
from keystoneauth1 import access
from keystoneauth1 import fixture
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit import utils
@staticmethod
def create_services(attrs=None, count=2):
    """Create multiple fake services.

        :param Dictionary attrs:
            A dictionary with all attributes
        :param int count:
            The number of services to fake
        :return:
            A list of FakeResource objects faking the services
        """
    services = []
    for i in range(0, count):
        services.append(FakeService.create_one_service(attrs))
    return services