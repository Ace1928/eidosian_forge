from concurrent import futures
from collections import namedtuple
import copy
import logging
import sys
import threading
from s3transfer.compat import MAXINT
from s3transfer.compat import six
from s3transfer.exceptions import CancelledError, TransferNotDoneError
from s3transfer.utils import FunctionContainer
from s3transfer.utils import TaskSemaphore
@property
def associated_futures(self):
    """The list of futures associated to the inprogress TransferFuture

        Once the transfer finishes this list becomes empty as the transfer
        is considered done and there should be no running futures left.
        """
    with self._associated_futures_lock:
        return copy.copy(self._associated_futures)