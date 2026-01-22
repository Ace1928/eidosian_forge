from __future__ import print_function
import sys
import os
import types
import traceback
from abc import abstractmethod
def check_namespace_char(ch):
    if u'!' <= ch <= u'~':
        return True
    if u'\xa0' <= ch <= u'\ud7ff':
        return True
    if u'\ue000' <= ch <= u'�' and ch != u'\ufeff':
        return True
    if u'𐀀' <= ch <= u'\U0010ffff':
        return True
    return False