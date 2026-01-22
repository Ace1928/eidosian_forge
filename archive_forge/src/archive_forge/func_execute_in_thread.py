from multiprocessing.pool import ThreadPool
from threading import Thread
from wandb_promise import Promise
from .utils import process
def execute_in_thread(self, fn, *args, **kwargs):
    promise = Promise()
    thread = Thread(target=process, args=(promise, fn, args, kwargs))
    thread.start()
    self.threads.append(thread)
    return promise