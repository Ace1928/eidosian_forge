from fontTools.ttLib import TTFont
from fontTools.varLib import models, VarLibError, load_designspace, load_masters
from fontTools.varLib.merger import InstancerMerger
import os.path
import logging
from copy import deepcopy
from pprint import pformat
Interpolate GDEF/GPOS/GSUB tables for a point on a designspace