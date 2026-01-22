import uuid
import fixtures
from openstackclient.tests.functional.image import base
def test_image_list_with_tag_filter(self):
    output = self.openstack('image list --tag ' + self.image_tag + ' --tag ' + self.image_tag1 + ' --long', parse_output=True)
    for taglist in [img['Tags'] for img in output]:
        self.assertIn(self.image_tag, taglist)
        self.assertIn(self.image_tag1, taglist)