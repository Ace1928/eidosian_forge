from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def use_pch(self, pch_sources, lang, inc_cl):
    pch = ET.SubElement(inc_cl, 'PrecompiledHeader')
    pch.text = 'Use'
    header = self.add_pch_files(pch_sources, lang, inc_cl)
    pch_include = ET.SubElement(inc_cl, 'ForcedIncludeFiles')
    pch_include.text = header + ';%(ForcedIncludeFiles)'