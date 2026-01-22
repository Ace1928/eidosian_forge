import datetime
import re
import sys
import threading
import time
import traceback
import unittest
from xml.sax import saxutils
from absl.testing import _pretty_print_reporter
@classmethod
def set_default_xml_stream(cls, xml_stream):
    """Sets the default XML stream for the class.

    Args:
      xml_stream: file-like or None; used for instances when xml_stream is None
          or not passed to their constructors.  If None is passed, instances
          created with xml_stream=None will act as ordinary TextTestRunner
          instances; this is the default state before any calls to this method
          have been made.
    """
    cls._xml_stream = xml_stream