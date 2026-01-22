from __future__ import absolute_import, division, unicode_literals
from six import text_type
import re
from copy import copy
from . import base
from .. import _ihatexml
from .. import constants
from ..constants import namespaces
from .._utils import moduleFactoryFactory
def reparentChildren(self, newParent):
    if newParent.childNodes:
        newParent.childNodes[-1]._element.tail += self._element.text
    else:
        if not newParent._element.text:
            newParent._element.text = ''
        if self._element.text is not None:
            newParent._element.text += self._element.text
    self._element.text = ''
    base.Node.reparentChildren(self, newParent)