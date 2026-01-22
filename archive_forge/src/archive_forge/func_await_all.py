from typing import List
from .types import AsyncTask, TaskStatus, WaitType, ErrorHandling
from .errors import AsyncTaskException
def await_all(tasks: List[AsyncTask], error_handling: ErrorHandling=ErrorHandling.RAISE):
    """
    Await all tasks to complete

    :param tasks: List of tasks
    :param error_handling: Error handling strategy (raise or ignore)
    """
    result = []
    for task in tasks:
        if task.status == TaskStatus.PENDING:
            while not task.thread.is_alive():
                pass
            task.thread.join()
    for task in tasks:
        if task.status == TaskStatus.SUCCESS:
            result.append(task.result)
        elif task.status == TaskStatus.FAILURE:
            if error_handling == ErrorHandling.RAISE:
                raise AsyncTaskException(str(task.exception), task.func.__name__)
    return result