import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def verbose_expired_key_message(result, repo) -> List[str]:
    """takes a verify result and returns list of expired key info"""
    signers: Dict[str, int] = {}
    fingerprint_to_authors = {}
    for rev_id, validity, fingerprint in result:
        if validity == SIGNATURE_EXPIRED:
            revision = repo.get_revision(rev_id)
            authors = ', '.join(revision.get_apparent_authors())
            signers.setdefault(fingerprint, 0)
            signers[fingerprint] += 1
            fingerprint_to_authors[fingerprint] = authors
    ret: List[str] = []
    for fingerprint, number in signers.items():
        ret.append(ngettext('{0} commit by author {1} with key {2} now expired', '{0} commits by author {1} with key {2} now expired', number).format(number, fingerprint_to_authors[fingerprint], fingerprint))
    return ret