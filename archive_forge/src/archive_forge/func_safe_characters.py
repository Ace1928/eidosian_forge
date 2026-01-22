from __future__ import absolute_import, unicode_literals
import sys
from . import SIGNATURE_METHODS, utils
@property
def safe_characters(self):
    return set(utils.UNICODE_ASCII_CHARACTER_SET)