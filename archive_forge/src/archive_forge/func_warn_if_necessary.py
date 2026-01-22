import time
import threading
def warn_if_necessary(tk_window, window_type=''):
    """
    When running within IPython, this function checks to see if a Tk event
    loop exists and, if not, tells the user how to start one.
    """
    try:
        import IPython
        ip = IPython.get_ipython()
        tk_window._have_loop = False

        def set_flag():
            tk_window._have_loop = True

        def tk_check():
            message = '\x1b[31mYour new {} window needs an event loop to become visible.\nType "%gui tk" below (without the quotes) to start one.\x1b[0m\n'.format(window_type if window_type else tk_window.winfo_class())
            if IPython.version_info < (6,):
                message = '\n' + message[:-1]
            for n in range(4):
                time.sleep(0.25)
                if tk_window._have_loop:
                    return
            print(message)
        tk_window.after(10, set_flag)
        threading.Thread(target=tk_check).start()
    except ImportError:
        pass