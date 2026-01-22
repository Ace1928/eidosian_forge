from . import handler
from ._exceptions import SAXNotSupportedException, SAXNotRecognizedException
def getErrorHandler(self):
    """Returns the current ErrorHandler."""
    return self._err_handler