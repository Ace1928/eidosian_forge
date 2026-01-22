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
def readRules(self):
    rules = []
    rulesElement = self.root.find('.rules')
    if rulesElement is not None:
        processingValue = rulesElement.attrib.get('processing', 'first')
        if processingValue not in {'first', 'last'}:
            raise DesignSpaceDocumentError("<rules> processing attribute value is not valid: %r, expected 'first' or 'last'" % processingValue)
        self.documentObject.rulesProcessingLast = processingValue == 'last'
    for ruleElement in self.root.findall('.rules/rule'):
        ruleObject = self.ruleDescriptorClass()
        ruleName = ruleObject.name = ruleElement.attrib.get('name')
        externalConditions = self._readConditionElements(ruleElement, ruleName)
        if externalConditions:
            ruleObject.conditionSets.append(externalConditions)
            self.log.info('Found stray rule conditions outside a conditionset. Wrapped them in a new conditionset.')
        for conditionSetElement in ruleElement.findall('.conditionset'):
            conditionSet = self._readConditionElements(conditionSetElement, ruleName)
            if conditionSet is not None:
                ruleObject.conditionSets.append(conditionSet)
        for subElement in ruleElement.findall('.sub'):
            a = subElement.attrib['name']
            b = subElement.attrib['with']
            ruleObject.subs.append((a, b))
        rules.append(ruleObject)
    self.documentObject.rules = rules