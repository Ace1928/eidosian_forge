import os
import sys
import uuid
import traceback
import json
import param
from ._version import __version__
@classmethod
def get_client_comm(cls, on_msg=None, id=None, on_error=None, on_stdout=None, on_open=None):
    comm = cls.client_comm(id, on_msg, on_error, on_stdout, on_open)
    cls._comms[comm.id] = comm
    return comm