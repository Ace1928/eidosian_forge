from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
@staticmethod
def to_branch(controldir):
    """Return a Reconfiguration to convert this controldir into a branch

        :param controldir: The controldir to reconfigure
        :raise AlreadyBranch: if controldir is already a branch
        """
    reconfiguration = Reconfigure(controldir)
    reconfiguration._plan_changes(want_tree=False, want_branch=True, want_bound=False, want_reference=False)
    if not reconfiguration.changes_planned():
        raise AlreadyBranch(controldir)
    return reconfiguration