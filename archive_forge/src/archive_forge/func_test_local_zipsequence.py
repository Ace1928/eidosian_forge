from os import remove
from os.path import join
from shutil import copyfile, rmtree
from tempfile import mkdtemp
from threading import Event
from zipfile import ZipFile
from kivy.tests.common import GraphicUnitTest, ensure_web_server
def test_local_zipsequence(self):
    from kivy import kivy_examples_dir
    zip_cube = join(kivy_examples_dir, 'widgets', 'sequenced_images', 'data', 'images', 'cube.zip')
    zip_pngs = self.zip_frames(zip_cube)
    image = self.load_zipimage(zip_cube, zip_pngs)
    self.assertTrue(self.check_sequence_frames(image._coreimage, int(image._coreimage.anim_delay * self.maxfps + 3)))