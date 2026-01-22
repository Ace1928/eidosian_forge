from kivy.uix.widget import Widget
from kivy.core.image import Image as CoreImage
from kivy.resources import resource_find
from kivy.properties import (
from kivy.logger import Logger
def on_nocache(self, *args):
    if self.nocache:
        self.remove_from_cache()
        if self._coreimage:
            self._coreimage._nocache = True