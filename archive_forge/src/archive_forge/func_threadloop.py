from queue import Queue, Empty
import threading
def threadloop(self):
    """Loops until all of the tasks are finished."""
    while True:
        args = self.queue.get()
        if args is STOP:
            self.queue.put(STOP)
            self.queue.task_done()
            break
        try:
            args[0](*args[1], **args[2])
        finally:
            self.queue.task_done()