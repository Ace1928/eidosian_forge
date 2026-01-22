import unittest
from tempfile import mkdtemp
from shutil import rmtree
@unittest.skip("Doesn't work with testsuite, but work alone")
def test_export_to_png(self):
    from kivy.core.image import Image as CoreImage
    from kivy.uix.button import Button
    from os.path import join
    wid = Button(text='test', size=(200, 100), size_hint=(None, None))
    self.root.add_widget(wid)
    tmp = mkdtemp()
    wid.export_to_png(join(tmp, 'a.png'))
    wid.export_to_png(join(tmp, 'b.png'), scale=0.5)
    wid.export_to_png(join(tmp, 'c.png'), scale=2)
    self.assertEqual(CoreImage(join(tmp, 'a.png')).size, (200, 100))
    self.assertEqual(CoreImage(join(tmp, 'b.png')).size, (100, 50))
    self.assertEqual(CoreImage(join(tmp, 'c.png')).size, (400, 200))
    rmtree(tmp)
    self.root.remove_widget(wid)