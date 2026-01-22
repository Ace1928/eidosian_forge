import random
import time
import functools
import math
import os
import socket
import stat
import string
import logging
import threading
import io
from collections import defaultdict
from botocore.exceptions import IncompleteReadError
from botocore.exceptions import ReadTimeoutError
from s3transfer.compat import SOCKET_ERROR
from s3transfer.compat import rename_file
from s3transfer.compat import seekable
from s3transfer.compat import fallocate
def get_callbacks(transfer_future, callback_type):
    """Retrieves callbacks from a subscriber

    :type transfer_future: s3transfer.futures.TransferFuture
    :param transfer_future: The transfer future the subscriber is associated
        to.

    :type callback_type: str
    :param callback_type: The type of callback to retrieve from the subscriber.
        Valid types include:
            * 'queued'
            * 'progress'
            * 'done'

    :returns: A list of callbacks for the type specified. All callbacks are
        preinjected with the transfer future.
    """
    callbacks = []
    for subscriber in transfer_future.meta.call_args.subscribers:
        callback_name = 'on_' + callback_type
        if hasattr(subscriber, callback_name):
            callbacks.append(functools.partial(getattr(subscriber, callback_name), future=transfer_future))
    return callbacks