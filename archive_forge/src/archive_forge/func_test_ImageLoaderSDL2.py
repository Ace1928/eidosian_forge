import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderSDL2(self):
    loadercls = LOADERS.get('ImageLoaderSDL2')
    if loadercls:
        exts = list(loadercls.extensions()) + ['gif']
        ctx = self._test_imageloader(loadercls, exts)