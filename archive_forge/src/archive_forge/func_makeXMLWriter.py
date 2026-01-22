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
def makeXMLWriter(newlinestr='\n'):
    writer = XMLWriter(BytesIO(), newlinestr=newlinestr)
    writer.file.seek(0)
    writer.file.truncate()
    return writer