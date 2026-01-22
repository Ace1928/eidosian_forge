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
def objgraph(self, type='Request', n=200, max_depth=10):
    """Create graph of uncollected objects (memory-leak debugging).

        Arguments:
            n (int): Max number of objects to graph.
            max_depth (int): Traverse at most n levels deep.
            type (str): Name of object to graph.  Default is ``"Request"``.

        Returns:
            Dict: Dictionary ``{'filename': FILENAME}``

        Note:
            Requires the objgraph library.
        """
    return self._request('objgraph', num=n, max_depth=max_depth, type=type)