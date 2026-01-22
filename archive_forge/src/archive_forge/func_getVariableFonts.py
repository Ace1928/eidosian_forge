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
def getVariableFonts(self) -> List[VariableFontDescriptor]:
    """Return all variable fonts defined in this document, or implicit
        variable fonts that can be built from the document's continuous axes.

        In the case of Designspace documents before version 5, the whole
        document was implicitly describing a variable font that covers the
        whole space.

        In version 5 and above documents, there can be as many variable fonts
        as there are locations on discrete axes.

        .. seealso:: :func:`splitInterpolable`

        .. versionadded:: 5.0
        """
    if self.variableFonts:
        return self.variableFonts
    variableFonts = []
    discreteAxes = []
    rangeAxisSubsets: List[Union[RangeAxisSubsetDescriptor, ValueAxisSubsetDescriptor]] = []
    for axis in self.axes:
        if hasattr(axis, 'values'):
            axis = cast(DiscreteAxisDescriptor, axis)
            discreteAxes.append(axis)
        else:
            rangeAxisSubsets.append(RangeAxisSubsetDescriptor(name=axis.name))
    valueCombinations = itertools.product(*[axis.values for axis in discreteAxes])
    for values in valueCombinations:
        basename = None
        if self.filename is not None:
            basename = os.path.splitext(self.filename)[0] + '-VF'
        if self.path is not None:
            basename = os.path.splitext(os.path.basename(self.path))[0] + '-VF'
        if basename is None:
            basename = 'VF'
        axisNames = ''.join([f'-{axis.tag}{value}' for axis, value in zip(discreteAxes, values)])
        variableFonts.append(VariableFontDescriptor(name=f'{basename}{axisNames}', axisSubsets=rangeAxisSubsets + [ValueAxisSubsetDescriptor(name=axis.name, userValue=value) for axis, value in zip(discreteAxes, values)]))
    return variableFonts