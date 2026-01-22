import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def get_elapsed_time(self):
    """Return the time that shows how long the test method took to
        execute.
        """
    return self.test_result.stop_time - self.test_result.start_time