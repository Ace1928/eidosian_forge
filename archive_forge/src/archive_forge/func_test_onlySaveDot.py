from __future__ import print_function
import functools
import os
import subprocess
from unittest import TestCase, skipIf
import attr
from .._methodical import MethodicalMachine
from .test_discover import isTwistedInstalled
def test_onlySaveDot(self):
    """
        Passing an empty string for --image-directory/-i disables
        rendering images.
        """
    for arg in ('--image-directory', '-i'):
        self.digraphRecorder.reset()
        self.collectedOutput = []
        self.tool(argv=[self.fakeFQPN, arg, ''])
        self.assertFalse(any(('image' in line for line in self.collectedOutput)))
        self.assertEqual(len(self.digraphRecorder.saveCalls), 1)
        call, = self.digraphRecorder.saveCalls
        self.assertEqual('{}.dot'.format(self.fakeFQPN), call['filename'])
        self.assertFalse(self.digraphRecorder.renderCalls)