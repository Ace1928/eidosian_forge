import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def commit_not_valid_message(count):
    """returns message for number of commits"""
    return ngettext('{0} commit not valid', '{0} commits not valid', count[SIGNATURE_NOT_VALID]).format(count[SIGNATURE_NOT_VALID])