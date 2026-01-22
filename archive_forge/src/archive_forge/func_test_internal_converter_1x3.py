import os
import re
import sys
import unittest
from collections import defaultdict
from kivy.core.image import ImageLoader
def test_internal_converter_1x3(self):
    pad5 = b'\x00' * 5
    correct = {'rgba': b'\x01\x02\x03\xff\x04\x05\x06\xff\x07\x08\t\xff', 'rgb_raw': b'\x01\x02\x03\x04\x05\x06\x07\x08\t', 'bgr_raw': b'\x03\x02\x01\x06\x05\x04\t\x08\x07', 'rgb_align2': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align2': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align4': b'\x01\x02\x03\x00\x04\x05\x06\x00\x07\x08\t\x00', 'bgr_align4': b'\x03\x02\x01\x00\x06\x05\x04\x00\t\x08\x07\x00', 'rgb_align8': b'\x01\x02\x03' + pad5 + b'\x04\x05\x06' + pad5 + b'\x07\x08\t' + pad5, 'bgr_align8': b'\x03\x02\x01' + pad5 + b'\x06\x05\x04' + pad5 + b'\t\x08\x07' + pad5}
    src = correct.get
    rgba = src('rgba')
    self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 4), src('rgb_align2'))
    self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 4), src('bgr_align2'))
    self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, None), src('rgb_align4'))
    self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, None), src('bgr_align4'))
    self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 0), src('rgb_raw'))
    self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 0), src('bgr_raw'))
    self.assertEqual(rgba_to(rgba, 'rgb', 1, 3, 8), src('rgb_align8'))
    self.assertEqual(rgba_to(rgba, 'bgr', 1, 3, 8), src('bgr_align8'))