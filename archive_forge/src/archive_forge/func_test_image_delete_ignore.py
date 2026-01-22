from openstack.image.v1 import _proxy
from openstack.image.v1 import image
from openstack.tests.unit import test_proxy_base as test_proxy_base
def test_image_delete_ignore(self):
    self.verify_delete(self.proxy.delete_image, image.Image, True)