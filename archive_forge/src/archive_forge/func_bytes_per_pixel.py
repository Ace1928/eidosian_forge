import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def bytes_per_pixel(fmt):
    if fmt in ('rgb', 'bgr'):
        return 3
    if fmt in ('rgba', 'bgra', 'argb', 'abgr'):
        return 4
    raise Exception('bytes_per_pixel: unknown format {}'.format(fmt))