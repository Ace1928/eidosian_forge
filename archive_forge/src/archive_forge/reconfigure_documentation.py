from . import branch, controldir, errors, trace, ui, urlutils
from .i18n import gettext
Apply the reconfiguration

        :param force: If true, the reconfiguration is applied even if it will
            destroy local changes.
        :raise errors.UncommittedChanges: if the local tree is to be destroyed
            but contains uncommitted changes.
        :raise NoBindLocation: if no bind location was specified and
            none could be autodetected.
        