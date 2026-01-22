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
def revoke_by_stamped_headers(self, headers, destination=None, terminate=False, signal=TERM_SIGNAME, **kwargs):
    """
        Tell all (or specific) workers to revoke a task by headers.

        If a task is revoked, the workers will ignore the task and
        not execute it after all.

        Arguments:
            headers (dict[str, Union(str, list)]): Headers to match when revoking tasks.
            terminate (bool): Also terminate the process currently working
                on the task (if any).
            signal (str): Name of signal to send to process if terminate.
                Default is TERM.

        See Also:
            :meth:`broadcast` for supported keyword arguments.
        """
    result = self.broadcast('revoke_by_stamped_headers', destination=destination, arguments={'headers': headers, 'terminate': terminate, 'signal': signal}, **kwargs)
    task_ids = set()
    if result:
        for host in result:
            for response in host.values():
                task_ids.update(response['ok'])
    if task_ids:
        return self.revoke(list(task_ids), destination=destination, terminate=terminate, signal=signal, **kwargs)
    else:
        return result