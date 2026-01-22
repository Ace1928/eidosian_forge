from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_image
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def setup_images_mock(self, count, servers=None):
    if servers:
        images = image_fakes.create_images(attrs={'name': servers[0].name, 'status': 'active'}, count=count)
    else:
        images = image_fakes.create_images(attrs={'status': 'active'}, count=count)
    self.image_client.find_image = mock.Mock(side_effect=images)
    self.compute_sdk_client.create_server_image = mock.Mock(return_value=images[0])
    return images