import time
import _tkinter
import tkinter
def wait_using_polling():
    """
        Windows TK doesn't support 'createfilehandler'.
        So, run the TK eventloop and poll until input is ready.
        """
    while not inputhook_context.input_is_ready():
        while root.dooneevent(_tkinter.ALL_EVENTS | _tkinter.DONT_WAIT):
            pass
        time.sleep(0.01)