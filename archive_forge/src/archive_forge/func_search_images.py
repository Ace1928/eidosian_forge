from openstack.cloud import _utils
from openstack import exceptions
from openstack.image.v2._proxy import Proxy
from openstack import utils
def search_images(self, name_or_id=None, filters=None):
    images = self.list_images()
    return _utils._filter_list(images, name_or_id, filters)