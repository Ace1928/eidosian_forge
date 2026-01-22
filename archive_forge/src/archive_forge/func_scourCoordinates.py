from __future__ import division         # use "true" division instead of integer division in Python 2 (see PEP 238)
from __future__ import print_function   # use print() as a function in Python 2 (see PEP 3105)
from __future__ import absolute_import  # use absolute imports by default in Python 2 (see PEP 328)
import math
import optparse
import os
import re
import sys
import time
import xml.dom.minidom
from xml.dom import Node, NotFoundErr
from collections import namedtuple, defaultdict
from decimal import Context, Decimal, InvalidOperation, getcontext
import six
from six.moves import range, urllib
from scour.svg_regex import svg_parser
from scour.svg_transform import svg_transform_parser
from scour.yocto_css import parseCssString
from scour import __version__
def scourCoordinates(data, options, force_whitespace=False, control_points=[], flags=[]):
    """
       Serializes coordinate data with some cleanups:
          - removes all trailing zeros after the decimal
          - integerize coordinates if possible
          - removes extraneous whitespace
          - adds spaces between values in a subcommand if required (or if force_whitespace is True)
    """
    if data is not None:
        newData = []
        c = 0
        previousCoord = ''
        for coord in data:
            is_control_point = c in control_points
            scouredCoord = scourUnitlessLength(coord, renderer_workaround=options.renderer_workaround, is_control_point=is_control_point)
            if c > 0 and (force_whitespace or scouredCoord[0].isdigit() or (scouredCoord[0] == '.' and (not ('.' in previousCoord or 'e' in previousCoord)))) and (c - 1 not in flags or options.renderer_workaround):
                newData.append(' ')
            newData.append(scouredCoord)
            previousCoord = scouredCoord
            c += 1
        if options.renderer_workaround:
            if len(newData) > 0:
                for i in range(1, len(newData)):
                    if newData[i][0] == '-' and 'e' in newData[i - 1]:
                        newData[i - 1] += ' '
                return ''.join(newData)
        else:
            return ''.join(newData)
    return ''