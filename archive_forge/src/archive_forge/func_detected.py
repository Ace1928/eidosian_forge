import unittest
import os
from jsbeautifier.unpackers.myobfuscate import detect, unpack
from jsbeautifier.unpackers.tests import __path__ as path
def detected(source):
    return self.assertTrue(detect(source))