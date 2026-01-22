import logging
import multiprocessing as mp
import os
import signal
import time
from queue import Empty
from typing import Any, Dict, List, Set, Tuple
from .api import RequestQueue, TimerClient, TimerRequest, TimerServer

    Server that works with ``LocalTimerClient``. Clients are expected to be
    subprocesses to the parent process that is running this server. Each host
    in the job is expected to start its own timer server locally and each
    server instance manages timers for local workers (running on processes
    on the same host).
    