import random
import unittest
import xmlrunner
from __future__ import absolute_import
import os
import sys
import time
from unittest import TestResult, TextTestResult, TextTestRunner
import xml.dom.minidom
def createCDATAOrText(self, data):
    if ']]>' in data:
        return self.createTextNode(data)
    return self.createCDATASection(data)