import os
import signal
import threading
import time
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from oslo_utils import strutils
from os_brick import exception
from os_brick import privileged
@privileged.default.entrypoint
def unlink_root(*links, **kwargs):
    """Unlink system links with sys admin privileges.

    By default it will raise an exception if a link does not exist and stop
    unlinking remaining links.

    This behavior can be modified passing optional parameters `no_errors` and
    `raise_at_end`.

    :param no_errors: Don't raise an exception on error
    "param raise_at_end: Don't raise an exception on first error, try to
                         unlink all links and then raise a ChainedException
                         with all the errors that where found.
    """
    no_errors = kwargs.get('no_errors', False)
    raise_at_end = kwargs.get('raise_at_end', False)
    exc = exception.ExceptionChainer()
    catch_exception = no_errors or raise_at_end
    LOG.debug('Unlinking %s', links)
    for link in links:
        with exc.context(catch_exception, 'Unlink failed for %s', link):
            os.unlink(link)
    if not no_errors and raise_at_end and exc:
        raise exc