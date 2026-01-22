import fnmatch
import locale
import os
import re
import stat
import subprocess
import sys
import textwrap
import types
import warnings
from xml.etree import ElementTree

        :return: the result of applying ``ElementTree.tostring()`` to
        the wrapped Element object.
        