from . import logger
import logging
def log_start(task, logger='TaskLogger'):
    """Begin logging of a task

    Convenience function to log a task in the default
    TaskLogger

    Parameters
    ----------
    task : str
        Name of the task to be started
    logger : str, optional (default: "TaskLogger")
        Unique name of the logger to retrieve

    Returns
    -------
    logger : TaskLogger
    """
    tasklogger = get_tasklogger(logger)
    tasklogger.start_task(task)
    return tasklogger