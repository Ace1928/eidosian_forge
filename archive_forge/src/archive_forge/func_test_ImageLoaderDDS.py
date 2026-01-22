import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_ImageLoaderDDS(self):
    loadercls = LOADERS.get('ImageLoaderDDS')
    ctx = self._test_imageloader(loadercls)