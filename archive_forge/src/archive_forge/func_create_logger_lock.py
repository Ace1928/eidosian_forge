import os
import threading
import weakref
def create_logger_lock():
    lock = threading.Lock()
    logger_locks.add(lock)
    return lock