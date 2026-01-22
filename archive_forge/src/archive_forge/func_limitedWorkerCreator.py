from queue import Queue
from threading import Lock, Thread, local as LocalStorage
from typing import Callable, Optional
from typing_extensions import Protocol
from twisted.python.log import err
from ._ithreads import IWorker
from ._team import Team
from ._threadworker import LockWorker, ThreadWorker
def limitedWorkerCreator() -> Optional[IWorker]:
    stats = team.statistics()
    if stats.busyWorkerCount + stats.idleWorkerCount >= currentLimit():
        return None
    return ThreadWorker(startThread, Queue())