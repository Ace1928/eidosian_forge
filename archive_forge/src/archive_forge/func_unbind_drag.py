from abc import ABCMeta, abstractmethod
from tkinter import (
from tkinter.filedialog import asksaveasfilename
from nltk.util import in_idle
def unbind_drag(self):
    """
        Remove a callback that was registered with ``bind_drag``.
        """
    try:
        del self.__callbacks['drag']
    except:
        pass