import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def get_pixel_alpha(pix, fmt):
    if fmt in ('rgba', 'bgra'):
        return pix[3]
    elif fmt in ('abgr', 'argb'):
        return pix[0]
    return 255