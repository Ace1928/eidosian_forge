import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_internal_converter_3x1(self):
    pad6 = b'\x00' * 6
    correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'abgr': b'\xff\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07', 'bgra': b'\x03\x02\x01\xff\x06\x05\x04\xff\t\x08\x07\xff', 'argb': b'\xff\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t', 'rgb_align2': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03\x04\x05\x06\x07\x08\t\x00' + pad6, 'bgr_align8': b'\x03\x02\x01\x06\x05\x04\t\x08\x07\x00' + pad6}
    src = correct.get
    rgba = src('rgba')
    self.assertEqual(rgba_to(rgba, 'bgra', 3, 1, 0), src('bgra'))
    self.assertEqual(rgba_to(rgba, 'argb', 3, 1, 0), src('argb'))
    self.assertEqual(rgba_to(rgba, 'abgr', 3, 1, 0), src('abgr'))
    self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 10), src('rgb_align2'))
    self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 10), src('bgr_align2'))
    self.assertEqual(rgba_to(rgba, 'rgb', 3, 1, 16), src('rgb_align8'))
    self.assertEqual(rgba_to(rgba, 'bgr', 3, 1, 16), src('bgr_align8'))