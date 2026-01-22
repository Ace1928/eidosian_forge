from kivy.tests.common import GraphicUnitTest
def test_image_properties(self):
    from kivy.uix.image import Image
    from os.path import dirname, join
    r = self.render
    filename = join(dirname(__file__), 'test_button.png')
    r(Image(source=filename))