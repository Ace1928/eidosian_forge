from __future__ import absolute_import, division, unicode_literals
import warnings
import re
import sys
from . import base
from ..constants import DataLossWarning
from .. import constants
from . import etree as etree_builders
from .. import _ihatexml
import lxml.etree as etree
from six import PY3, binary_type
def insertCommentMain(self, data, parent=None):
    if parent == self.document and self.document._elementTree.getroot()[-1].tag == comment_type:
        warnings.warn('lxml cannot represent adjacent comments beyond the root elements', DataLossWarning)
    super(TreeBuilder, self).insertComment(data, parent)