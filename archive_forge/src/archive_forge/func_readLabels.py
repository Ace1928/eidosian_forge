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
def readLabels(self):
    if self.documentObject.formatTuple < (5, 0):
        return
    xml_attrs = {'name', 'elidable', 'oldersibling'}
    for labelElement in self.root.findall('.labels/label'):
        unknown_attrs = set(labelElement.attrib) - xml_attrs
        if unknown_attrs:
            raise DesignSpaceDocumentError(f'Label element contains unknown attributes: {', '.join(unknown_attrs)}')
        name = labelElement.get('name')
        if name is None:
            raise DesignSpaceDocumentError('label element must have a name attribute.')
        designLocation, userLocation = self.locationFromElement(labelElement)
        if designLocation:
            raise DesignSpaceDocumentError(f'<label> element "{name}" must only have user locations (using uservalue="").')
        elidable = True if labelElement.get('elidable') == 'true' else False
        olderSibling = True if labelElement.get('oldersibling') == 'true' else False
        labelNames = {lang: label_name.text or '' for label_name in labelElement.findall('labelname') for attr, lang in label_name.items() if attr == XML_LANG}
        locationLabel = self.locationLabelDescriptorClass(name=name, userLocation=userLocation, elidable=elidable, olderSibling=olderSibling, labelNames=labelNames)
        self.documentObject.locationLabels.append(locationLabel)