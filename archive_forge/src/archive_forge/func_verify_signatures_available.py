import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
@staticmethod
def verify_signatures_available():
    """
        check if this strategy can verify signatures

        :return: boolean if this strategy can verify signatures
        """
    try:
        import gpg
        return True
    except ModuleNotFoundError:
        return False