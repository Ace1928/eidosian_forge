import os
from concurrent.futures import _base
import queue
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing.queues import Queue
import threading
import weakref
from functools import partial
import itertools
import sys
from traceback import format_exception
def process_result_item(self, result_item):
    if isinstance(result_item, int):
        assert self.is_shutting_down()
        p = self.processes.pop(result_item)
        p.join()
        if not self.processes:
            self.join_executor_internals()
            return
    else:
        work_item = self.pending_work_items.pop(result_item.work_id, None)
        if work_item is not None:
            if result_item.exception:
                work_item.future.set_exception(result_item.exception)
            else:
                work_item.future.set_result(result_item.result)