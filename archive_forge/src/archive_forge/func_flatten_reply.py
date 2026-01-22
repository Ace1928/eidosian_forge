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
def flatten_reply(reply):
    """Flatten node replies.

    Convert from a list of replies in this format::

        [{'a@example.com': reply},
         {'b@example.com': reply}]

    into this format::

        {'a@example.com': reply,
         'b@example.com': reply}
    """
    nodes, dupes = ({}, set())
    for item in reply:
        [dupes.add(name) for name in item if name in nodes]
        nodes.update(item)
    if dupes:
        warnings.warn(DuplicateNodenameWarning(W_DUPNODE.format(pluralize(len(dupes), 'name'), ', '.join(sorted(dupes)))))
    return nodes