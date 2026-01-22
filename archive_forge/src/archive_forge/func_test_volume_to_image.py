from fixtures import TimeoutException
from testtools import content
from openstack import exceptions
from openstack.tests.functional import base
from openstack import utils
def test_volume_to_image(self):
    """Test volume export to image functionality"""
    volume_name = self.getUniqueString()
    image_name = self.getUniqueString()
    self.addDetail('volume', content.text_content(volume_name))
    self.addCleanup(self.cleanup, volume_name, image_name=image_name)
    volume = self.user_cloud.create_volume(display_name=volume_name, size=1)
    image = self.user_cloud.create_image(image_name, volume=volume, wait=True)
    volume_ids = [v['id'] for v in self.user_cloud.list_volumes()]
    self.assertIn(volume['id'], volume_ids)
    image_list = self.user_cloud.list_images()
    image_ids = [s['id'] for s in image_list]
    self.assertIn(image['id'], image_ids)
    self.user_cloud.delete_image(image_name, wait=True)
    self.user_cloud.delete_volume(volume_name, wait=True)