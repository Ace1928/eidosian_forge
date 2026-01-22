import copy
import glob
import inspect
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime
from numbers import Number
from threading import Thread
from typing import Any, Callable, Dict, List, Optional, Sequence, Type, Union
import numpy as np
import psutil
import ray
from ray.util.annotations import DeveloperAPI, PublicAPI
from ray.air._internal.json import SafeFallbackEncoder  # noqa
from ray.air._internal.util import (  # noqa: F401
from ray._private.dict import (  # noqa: F401
@DeveloperAPI
def retry_fn(fn: Callable[[], Any], exception_type: Union[Type[Exception], Sequence[Type[Exception]]]=Exception, num_retries: int=3, sleep_time: int=1, timeout: Optional[Number]=None) -> bool:
    errored = threading.Event()

    def _try_fn():
        try:
            fn()
        except exception_type as e:
            logger.warning(e)
            errored.set()
    for i in range(num_retries):
        errored.clear()
        proc = threading.Thread(target=_try_fn)
        proc.daemon = True
        proc.start()
        proc.join(timeout=timeout)
        if proc.is_alive():
            logger.debug(f'Process timed out (try {i + 1}/{num_retries}): {getattr(fn, '__name__', None)}')
        elif not errored.is_set():
            return True
        time.sleep(sleep_time)
    return False