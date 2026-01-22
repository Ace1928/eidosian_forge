import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def verbose_missing_key_message(result) -> List[str]:
    """takes a verify result and returns list of missing key info"""
    signers: Dict[str, int] = {}
    for rev_id, validity, fingerprint in result:
        if validity == SIGNATURE_KEY_MISSING:
            signers.setdefault(fingerprint, 0)
            signers[fingerprint] += 1
    ret: List[str] = []
    for fingerprint, number in list(signers.items()):
        ret.append(ngettext('Unknown key {0} signed {1} commit', 'Unknown key {0} signed {1} commits', number).format(fingerprint, number))
    return ret