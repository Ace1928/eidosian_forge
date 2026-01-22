import filecmp
import os
import tempfile
from openstack.tests.functional import base
def test_create_image_force_duplicate(self):
    test_image = tempfile.NamedTemporaryFile(delete=False)
    test_image.write(b'\x00' * 1024 * 1024)
    test_image.close()
    image_name = self.getUniqueString('image')
    first_image = None
    second_image = None
    try:
        first_image = self.user_cloud.create_image(name=image_name, filename=test_image.name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, wait=True)
        second_image = self.user_cloud.create_image(name=image_name, filename=test_image.name, disk_format='raw', container_format='bare', min_disk=10, min_ram=1024, allow_duplicates=True, wait=True)
        self.assertNotEqual(first_image.id, second_image.id)
    finally:
        if first_image:
            self.user_cloud.delete_image(first_image.id, wait=True)
        if second_image:
            self.user_cloud.delete_image(second_image.id, wait=True)