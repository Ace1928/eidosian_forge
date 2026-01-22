import sys
import os
import os.path
import tempfile
import zipfile
from xml.dom import minidom
import time
import re
import copy
import itertools
import docutils
from docutils import frontend, nodes, utils, writers, languages
from docutils.readers import standalone
from docutils.transforms import references
def write_zip_str(self, zfile, name, bytes, compress_type=zipfile.ZIP_DEFLATED):
    localtime = time.localtime(time.time())
    zinfo = zipfile.ZipInfo(name, localtime)
    zinfo.external_attr = (33188 & 65535) << 16
    zinfo.compress_type = compress_type
    zfile.writestr(zinfo, bytes)