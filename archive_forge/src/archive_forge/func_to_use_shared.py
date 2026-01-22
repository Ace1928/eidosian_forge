from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
@classmethod
def to_use_shared(klass, controldir):
    """Convert a standalone branch into a repository branch"""
    reconfiguration = klass(controldir)
    reconfiguration._set_use_shared(use_shared=True)
    if not reconfiguration.changes_planned():
        raise AlreadyUsingShared(controldir)
    return reconfiguration