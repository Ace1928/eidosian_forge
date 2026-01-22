from __future__ import annotations
import collections
import copy
import itertools
import math
import os
import posixpath
from io import BytesIO, StringIO
from textwrap import indent
from typing import Any, Dict, List, MutableMapping, Optional, Tuple, Union, cast
from fontTools.misc import etree as ET
from fontTools.misc import plistlib
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.textTools import tobytes, tostr
def readLocationElement(self, locationElement):
    """Read a ``<location>`` element.

        .. versionchanged:: 5.0
           Return a tuple of (designLocation, userLocation)
        """
    if self._strictAxisNames and (not self.documentObject.axes):
        raise DesignSpaceDocumentError('No axes defined')
    userLoc = {}
    designLoc = {}
    for dimensionElement in locationElement.findall('.dimension'):
        dimName = dimensionElement.attrib.get('name')
        if self._strictAxisNames and dimName not in self.axisDefaults:
            self.log.warning('Location with undefined axis: "%s".', dimName)
            continue
        userValue = xValue = yValue = None
        try:
            userValue = dimensionElement.attrib.get('uservalue')
            if userValue is not None:
                userValue = float(userValue)
        except ValueError:
            self.log.warning('ValueError in readLocation userValue %3.3f', userValue)
        try:
            xValue = dimensionElement.attrib.get('xvalue')
            if xValue is not None:
                xValue = float(xValue)
        except ValueError:
            self.log.warning('ValueError in readLocation xValue %3.3f', xValue)
        try:
            yValue = dimensionElement.attrib.get('yvalue')
            if yValue is not None:
                yValue = float(yValue)
        except ValueError:
            self.log.warning('ValueError in readLocation yValue %3.3f', yValue)
        if userValue is None == xValue is None:
            raise DesignSpaceDocumentError(f'Exactly one of uservalue="" or xvalue="" must be provided for location dimension "{dimName}"')
        if yValue is not None:
            if xValue is None:
                raise DesignSpaceDocumentError(f'Missing xvalue="" for the location dimension "{dimName}"" with yvalue="{yValue}"')
            designLoc[dimName] = (xValue, yValue)
        elif xValue is not None:
            designLoc[dimName] = xValue
        else:
            userLoc[dimName] = userValue
    return (designLoc, userLoc)