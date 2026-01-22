import contextlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from traits.etsconfig.etsconfig import ETSConfig, ETSToolkitError
def test_toolkit_default_kiva_backend(self):
    self.ETSConfig.toolkit = 'qt4'
    self.assertEqual(self.ETSConfig.kiva_backend, 'image')