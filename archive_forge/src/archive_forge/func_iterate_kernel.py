import sys
import warnings
import gobject
import gtk
def iterate_kernel(self):
    """Run one iteration of the kernel and return True.

        GTK timer functions must return True to be called again, so we make the
        call to :meth:`do_one_iteration` and then return True for GTK.
        """
    self.kernel.do_one_iteration()
    return True