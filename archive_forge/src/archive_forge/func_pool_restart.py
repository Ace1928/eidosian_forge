import warnings
from billiard.common import TERM_SIGNAME
from kombu.matcher import match
from kombu.pidbox import Mailbox
from kombu.utils.compat import register_after_fork
from kombu.utils.functional import lazy
from kombu.utils.objects import cached_property
from celery.exceptions import DuplicateNodenameWarning
from celery.utils.log import get_logger
from celery.utils.text import pluralize
def pool_restart(self, modules=None, reload=False, reloader=None, destination=None, **kwargs):
    """Restart the execution pools of all or specific workers.

        Keyword Arguments:
            modules (Sequence[str]): List of modules to reload.
            reload (bool): Flag to enable module reloading.  Default is False.
            reloader (Any): Function to reload a module.
            destination (Sequence[str]): List of worker names to send this
                command to.

        See Also:
            Supports the same arguments as :meth:`broadcast`
        """
    return self.broadcast('pool_restart', arguments={'modules': modules, 'reload': reload, 'reloader': reloader}, destination=destination, **kwargs)