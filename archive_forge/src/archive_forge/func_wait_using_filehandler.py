import time
import _tkinter
import tkinter
def wait_using_filehandler():
    """
        Run the TK eventloop until the file handler that we got from the
        inputhook becomes readable.
        """
    stop = [False]

    def done(*a):
        stop[0] = True
    root.createfilehandler(inputhook_context.fileno(), _tkinter.READABLE, done)
    while root.dooneevent(_tkinter.ALL_EVENTS):
        if stop[0]:
            break
    root.deletefilehandler(inputhook_context.fileno())