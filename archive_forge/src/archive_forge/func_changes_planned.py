from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
def changes_planned(self):
    """Return True if changes are planned, False otherwise"""
    return self._unbind or self._bind or self._destroy_tree or self._create_tree or self._destroy_reference or self._create_branch or self._create_repository or self._create_reference or self._destroy_repository