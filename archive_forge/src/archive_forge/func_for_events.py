import select
from pyudev._util import eintr_retry_call
@classmethod
def for_events(cls, *events):
    """Listen for ``events``.

        ``events`` is a list of ``(fd, event)`` pairs, where ``fd`` is a file
        descriptor or file object and ``event`` either ``'r'`` or ``'w'``.  If
        ``r``, listen for whether that is ready to be read.  If ``w``, listen
        for whether the channel is ready to be written to.

        """
    notifier = eintr_retry_call(select.poll)
    for fd, event in events:
        mask = cls._EVENT_TO_MASK.get(event)
        if not mask:
            raise ValueError(f'Unknown event type: {repr(event)}')
        notifier.register(fd, mask)
    return cls(notifier)