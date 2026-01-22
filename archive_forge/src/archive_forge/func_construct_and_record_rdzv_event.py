import inspect
import logging
import os
import socket
import traceback
from enum import Enum
from typing import Dict, Optional
from torch.distributed.elastic.events.handlers import get_logging_handler
from .api import (  # noqa: F401
def construct_and_record_rdzv_event(run_id: str, message: str, node_state: NodeState, name: str='', hostname: str='', pid: Optional[int]=None, master_endpoint: str='', local_id: Optional[int]=None, rank: Optional[int]=None) -> None:
    if isinstance(get_logging_handler('dynamic_rendezvous'), logging.NullHandler):
        return
    if not hostname:
        hostname = socket.getfqdn()
    if not pid:
        pid = os.getpid()
    callstack = inspect.stack()
    filename = 'no_file'
    if len(callstack) > 1:
        stack_depth_1 = callstack[1]
        filename = os.path.basename(stack_depth_1.filename)
        if not name:
            name = stack_depth_1.function
    del callstack
    if node_state == NodeState.FAILED:
        error_trace = traceback.format_exc()
    else:
        error_trace = ''
    event = RdzvEvent(name=f'{filename}:{name}', run_id=run_id, message=message, hostname=hostname, pid=pid, node_state=node_state, master_endpoint=master_endpoint, rank=rank, local_id=local_id, error_trace=error_trace)
    record_rdzv_event(event)