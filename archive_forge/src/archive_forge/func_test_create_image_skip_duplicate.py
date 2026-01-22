import filecmp
import os
import tempfile
from openstack.tests.functional import base
def test_create_image_skip_duplicate(self):
    test_image = tempfile.NamedTemporaryFile(delete=False)
    test_image.write(b'\x00' * 1024 * 1024)
    test_image.close()
    image_name = self.getUniqueString('image')
    try:
        first_image = self.user_cloud.create_image(name=image_name, filename=test_image.name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, validate_checksum=True, wait=True)
        second_image = self.user_cloud.create_image(name=image_name, filename=test_image.name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, validate_checksum=True, wait=True)
        self.assertEqual(first_image.id, second_image.id)
    finally:
        self.user_cloud.delete_image(image_name, wait=True)