import time
import threading
def tk_check():
    message = '\x1b[31mYour new {} window needs an event loop to become visible.\nType "%gui tk" below (without the quotes) to start one.\x1b[0m\n'.format(window_type if window_type else tk_window.winfo_class())
    if IPython.version_info < (6,):
        message = '\n' + message[:-1]
    for n in range(4):
        time.sleep(0.25)
        if tk_window._have_loop:
            return
    print(message)