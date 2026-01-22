from libcloud.container.base import ContainerImage
def test_list_images_response(self):
    images = self.driver.list_images()
    self.assertTrue(isinstance(images, list))
    for image in images:
        self.assertTrue(isinstance(image, ContainerImage))