import os
from typing import Dict, List, Optional
from . import config, errors, trace, ui
from .i18n import gettext, ngettext
def verbose_not_signed_message(result, repo) -> List[str]:
    """takes a verify result and returns list of not signed commit info"""
    signers: Dict[str, int] = {}
    for rev_id, validity, empty in result:
        if validity == SIGNATURE_NOT_SIGNED:
            revision = repo.get_revision(rev_id)
            authors = ', '.join(revision.get_apparent_authors())
            signers.setdefault(authors, 0)
            signers[authors] += 1
    ret: List[str] = []
    for authors, number in signers.items():
        ret.append(ngettext('{0} commit by author {1}', '{0} commits by author {1}', number).format(number, authors))
    return ret