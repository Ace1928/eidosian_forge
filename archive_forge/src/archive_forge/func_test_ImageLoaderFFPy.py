import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderFFPy(self):
    loadercls = LOADERS.get('ImageLoaderFFPy')
    ctx = self._test_imageloader(loadercls)