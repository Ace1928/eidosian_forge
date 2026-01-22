import time
import asyncio
import threading
from lazyops.utils.logs import logger
from typing import List, Union, Set, Callable, Tuple, Optional
def spawn_server_tasks():
    """
    Spawn server tasks
    """
    proc = threading.Thread(target=start_bg_tasks)
    proc.start()
    while True:
        try:
            time.sleep(5.0)
        except Exception as e:
            logger.info('Stopping server tasks...')
            for task in _server_tasks:
                task.cancel()
            break
    proc.join()