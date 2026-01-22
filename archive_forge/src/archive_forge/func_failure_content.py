from typing import List, TYPE_CHECKING
from functools import partial
from testtools.content import TracebackContent
def failure_content(failure):
    """Create a Content object for a Failure.

    :param Failure failure: The failure to create content for.
    :rtype: ``Content``
    """
    return TracebackContent((failure.type, failure.value, failure.getTracebackObject()), None)