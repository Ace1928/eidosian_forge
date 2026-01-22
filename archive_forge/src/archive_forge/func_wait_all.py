import secrets
import string
import threading
import time
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple
from wandb.errors import Error
from wandb.proto import wandb_internal_pb2 as pb
def wait_all(self, handles: List[MailboxHandle], *, timeout: float, on_progress_all: Optional[Callable[[MailboxProgressAll], None]]=None) -> bool:
    progress_all_handle: Optional[MailboxProgressAll] = None
    if on_progress_all:
        progress_all_handle = MailboxProgressAll()
    wait_all = _MailboxWaitAll()
    for handle in handles:
        wait_all._add_handle(handle)
        if progress_all_handle and handle._on_progress:
            progress_handle = MailboxProgress(_handle=handle)
            if handle._on_probe:
                probe_handle = MailboxProbe()
                progress_handle.add_probe_handle(probe_handle)
            progress_all_handle.add_progress_handle(progress_handle)
    start_time = self._time()
    while wait_all.active_handles_count > 0:
        if self._keepalive:
            for handle in wait_all.active_handles:
                if not handle._interface:
                    continue
                if handle._interface._transport_keepalive_failed():
                    wait_all._mark_handle_failed(handle)
            if not wait_all.active_handles_count:
                if wait_all.failed_handles_count:
                    wait_all.clear_handles()
                    raise MailboxError('transport failed')
                break
        wait_all._get_and_clear(timeout=1)
        if progress_all_handle and on_progress_all:
            for progress_handle in progress_all_handle.get_progress_handles():
                for probe_handle in progress_handle.get_probe_handles():
                    if progress_handle._handle and progress_handle._handle._on_probe:
                        progress_handle._handle._on_probe(probe_handle)
            on_progress_all(progress_all_handle)
        now = self._time()
        if timeout >= 0 and now >= start_time + timeout:
            break
    return wait_all.active_handles_count == 0