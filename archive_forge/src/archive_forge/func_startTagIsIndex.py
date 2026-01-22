from __future__ import absolute_import, division, unicode_literals
from six import with_metaclass, viewkeys
import types
from . import _inputstream
from . import _tokenizer
from . import treebuilders
from .treebuilders.base import Marker
from . import _utils
from .constants import (
def startTagIsIndex(self, token):
    self.parser.parseError('deprecated-tag', {'name': 'isindex'})
    if self.tree.formPointer:
        return
    form_attrs = {}
    if 'action' in token['data']:
        form_attrs['action'] = token['data']['action']
    self.processStartTag(impliedTagToken('form', 'StartTag', attributes=form_attrs))
    self.processStartTag(impliedTagToken('hr', 'StartTag'))
    self.processStartTag(impliedTagToken('label', 'StartTag'))
    if 'prompt' in token['data']:
        prompt = token['data']['prompt']
    else:
        prompt = 'This is a searchable index. Enter search keywords: '
    self.processCharacters({'type': tokenTypes['Characters'], 'data': prompt})
    attributes = token['data'].copy()
    if 'action' in attributes:
        del attributes['action']
    if 'prompt' in attributes:
        del attributes['prompt']
    attributes['name'] = 'isindex'
    self.processStartTag(impliedTagToken('input', 'StartTag', attributes=attributes, selfClosing=token['selfClosing']))
    self.processEndTag(impliedTagToken('label'))
    self.processStartTag(impliedTagToken('hr', 'StartTag'))
    self.processEndTag(impliedTagToken('form'))