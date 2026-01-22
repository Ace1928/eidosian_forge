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
def optimizeTransform(transform):
    """
    Optimises a series of transformations parsed from a single
    transform="" attribute.

    The transformation list is modified in-place.
    """
    if len(transform) == 1 and transform[0][0] == 'matrix':
        matrix = A1, B1, A2, B2, A3, B3 = transform[0][1]
        if matrix == [1, 0, 0, 1, 0, 0]:
            del transform[0]
        elif A1 == 1 and A2 == 0 and (B1 == 0) and (B2 == 1):
            transform[0] = ('translate', [A3, B3])
        elif A2 == 0 and A3 == 0 and (B1 == 0) and (B3 == 0):
            transform[0] = ('scale', [A1, B2])
        elif A1 == B2 and -1 <= A1 <= 1 and (A3 == 0) and (-B1 == A2) and (-1 <= B1 <= 1) and (B3 == 0) and (abs(B1 ** 2 + A1 ** 2 - 1) < Decimal('1e-15')):
            sin_A, cos_A = (B1, A1)
            A = Decimal(str(math.degrees(math.asin(float(sin_A)))))
            if cos_A < 0:
                if sin_A < 0:
                    A = -180 - A
                else:
                    A = 180 - A
            transform[0] = ('rotate', [A])
    for type, args in transform:
        if type == 'translate':
            if len(args) == 2 and args[1] == 0:
                del args[1]
        elif type == 'rotate':
            args[0] = optimizeAngle(args[0])
            if len(args) == 3 and args[1] == args[2] == 0:
                del args[1:]
        elif type == 'scale':
            if len(args) == 2 and args[0] == args[1]:
                del args[1]
    i = 1
    while i < len(transform):
        currType, currArgs = transform[i]
        prevType, prevArgs = transform[i - 1]
        if currType == prevType == 'translate':
            prevArgs[0] += currArgs[0]
            if len(currArgs) == 2:
                if len(prevArgs) == 2:
                    prevArgs[1] += currArgs[1]
                elif len(prevArgs) == 1:
                    prevArgs.append(currArgs[1])
            del transform[i]
            if prevArgs[0] == prevArgs[1] == 0:
                i -= 1
                del transform[i]
        elif currType == prevType == 'rotate' and len(prevArgs) == len(currArgs) == 1:
            prevArgs[0] = optimizeAngle(prevArgs[0] + currArgs[0])
            del transform[i]
        elif currType == prevType == 'scale':
            prevArgs[0] *= currArgs[0]
            if len(prevArgs) == 2 and len(currArgs) == 2:
                prevArgs[1] *= currArgs[1]
            elif len(prevArgs) == 1 and len(currArgs) == 2:
                prevArgs.append(prevArgs[0] * currArgs[1])
            elif len(prevArgs) == 2 and len(currArgs) == 1:
                prevArgs[1] *= currArgs[0]
            del transform[i]
            if prevArgs[0] == 1 and (len(prevArgs) == 1 or prevArgs[1] == 1):
                i -= 1
                del transform[i]
        else:
            i += 1
    i = 0
    while i < len(transform):
        currType, currArgs = transform[i]
        if (currType == 'skewX' or currType == 'skewY') and len(currArgs) == 1 and (currArgs[0] == 0):
            del transform[i]
        elif currType == 'rotate' and len(currArgs) == 1 and (currArgs[0] == 0):
            del transform[i]
        else:
            i += 1