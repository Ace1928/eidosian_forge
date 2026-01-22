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
def revoked(self):
    """Return list of revoked tasks.

        >>> app.control.inspect().revoked()
        {'celery@node1': ['16f527de-1c72-47a6-b477-c472b92fef7a']}

        Returns:
            Dict: Dictionary ``{HOSTNAME: [TASK_ID, ...]}``.
        """
    return self._request('revoked')