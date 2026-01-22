from collections.abc import Iterable
from io import BytesIO
import os
import re
import shutil
import sys
import tempfile
from unittest import TestCase as _TestCase
from fontTools.config import Config
from fontTools.misc.textTools import tobytes
from fontTools.misc.xmlWriter import XMLWriter
def startElement_(self, name, attrs):
    element = (name, attrs, [])
    if self.stack:
        self.stack[-1][2].append(element)
    else:
        self.root = element
    self.stack.append(element)