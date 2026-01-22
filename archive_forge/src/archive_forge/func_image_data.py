from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils as common_utils
from openstackclient.compute.v2 import server_image
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
def image_data(self, image):
    datalist = (image['id'], image['name'], image['owner_id'], image['is_protected'], 'active', format_columns.ListColumn(image.get('tags')), image['visibility'])
    return datalist