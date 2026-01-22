import commands
import difflib
import getpass
import itertools
import os
import re
import subprocess
import sys
import tempfile
import types
from google.apputils import app
import gflags as flags
from google.apputils import shellutil
def recordProperty(self, property_name, property_value):
    """Record an arbitrary property for later use.

    Args:
      property_name: str, name of property to record; must be a valid XML
        attribute name
      property_value: value of property; must be valid XML attribute value
    """
    self.__recorded_properties[property_name] = property_value