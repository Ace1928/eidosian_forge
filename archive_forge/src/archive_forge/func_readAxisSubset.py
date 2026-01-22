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
def readAxisSubset(self, element: ET.Element):
    if 'uservalue' in element.attrib:
        xml_attrs = {'name', 'uservalue'}
        unknown_attrs = set(element.attrib) - xml_attrs
        if unknown_attrs:
            raise DesignSpaceDocumentError(f'axis-subset element contains unknown attributes: {', '.join(unknown_attrs)}')
        name = element.get('name')
        if name is None:
            raise DesignSpaceDocumentError('axis-subset element must have a name attribute.')
        userValueStr = element.get('uservalue')
        if userValueStr is None:
            raise DesignSpaceDocumentError('The axis-subset element for a discrete subset must have a uservalue attribute.')
        userValue = float(userValueStr)
        return self.valueAxisSubsetDescriptorClass(name=name, userValue=userValue)
    else:
        xml_attrs = {'name', 'userminimum', 'userdefault', 'usermaximum'}
        unknown_attrs = set(element.attrib) - xml_attrs
        if unknown_attrs:
            raise DesignSpaceDocumentError(f'axis-subset element contains unknown attributes: {', '.join(unknown_attrs)}')
        name = element.get('name')
        if name is None:
            raise DesignSpaceDocumentError('axis-subset element must have a name attribute.')
        userMinimum = element.get('userminimum')
        userDefault = element.get('userdefault')
        userMaximum = element.get('usermaximum')
        if userMinimum is not None and userDefault is not None and (userMaximum is not None):
            return self.rangeAxisSubsetDescriptorClass(name=name, userMinimum=float(userMinimum), userDefault=float(userDefault), userMaximum=float(userMaximum))
        if all((v is None for v in (userMinimum, userDefault, userMaximum))):
            return self.rangeAxisSubsetDescriptorClass(name=name)
        raise DesignSpaceDocumentError('axis-subset element must have min/max/default values or none at all.')