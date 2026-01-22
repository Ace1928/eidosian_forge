from _pydev_bundle._pydev_saved_modules import threading
def wrap_threads():
    threading.Lock = factory_wrapper(threading.Lock)
    threading.RLock = factory_wrapper(threading.RLock)
    import queue
    queue.Queue = factory_wrapper(queue.Queue)