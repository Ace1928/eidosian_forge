from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_saveDotAndImagesInDifferentDirectories(self):
    """
        Passing different directories to --image-directory and --dot-directory
        writes images and dot files to those directories.
        """
    imageDirectory = 'image'
    dotDirectory = 'dot'
    self.tool(argv=[self.fakeFQPN, '--image-directory', imageDirectory, '--dot-directory', dotDirectory])
    self.assertTrue(any(('image' in line for line in self.collectedOutput)))
    self.assertTrue(any(('dot' in line for line in self.collectedOutput)))
    self.assertEqual(len(self.digraphRecorder.renderCalls), 1)
    renderCall, = self.digraphRecorder.renderCalls
    self.assertEqual(renderCall['directory'], imageDirectory)
    self.assertTrue(renderCall['cleanup'])
    self.assertEqual(len(self.digraphRecorder.saveCalls), 1)
    saveCall, = self.digraphRecorder.saveCalls
    self.assertEqual(saveCall['directory'], dotDirectory)