from typing import TYPE_CHECKING
from .queue import Queue
from .utils import as_text

    Check whether there are any jobs stuck in the intermediate queue.

    A job may be stuck in the intermediate queue if a worker has successfully dequeued a job
    but was not able to push it to the StartedJobRegistry. This may happen in rare cases
    of hardware or network failure.

    We consider a job to be stuck in the intermediate queue if it doesn't exist in the StartedJobRegistry.
    