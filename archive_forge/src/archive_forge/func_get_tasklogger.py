from . import logger
import logging
def get_tasklogger(name='TaskLogger'):
    """Get a TaskLogger object

    Parameters
    ----------
    logger : str, optional (default: "TaskLogger")
        Unique name of the logger to retrieve

    Returns
    -------
    logger : TaskLogger
    """
    try:
        return logging.getLogger(name).tasklogger
    except AttributeError:
        return logger.TaskLogger(name)