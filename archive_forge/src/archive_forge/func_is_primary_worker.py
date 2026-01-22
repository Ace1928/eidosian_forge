import builtins
import copy
import os
import pickle
import contextlib
import subprocess
import socket
import parlai.utils.logging as logging
def is_primary_worker():
    """
    Determine if we are the primary (rank 0)  worker.

    Returns False if we are a secondary worker. Returns True if we are either (1) not in
    distributed mode (2) or are the primary (rank 0) worker.
    """
    return not is_distributed() or dist.get_rank() == 0