from os import remove, rmdir, mkdir
from os.path import join, dirname, isdir
import unittest
from zipfile import ZipFile
import pytest
from kivy.clock import Clock
from kivy.uix.filechooser import FileChooserListView
from kivy.utils import platform
@pytest.mark.skipif(platform == 'macosx' or platform == 'ios', reason='Unicode files unpredictable on MacOS and iOS')
def test_filechooserlistview_unicode(self):
    wid = FileChooserListView(path=self.subdir)
    Clock.tick()
    files = [join(self.subdir, f) for f in wid.files]
    for f in self.files:
        self.assertIn(f, files)
    for f in self.existfiles:
        self.assertIn(f, files)