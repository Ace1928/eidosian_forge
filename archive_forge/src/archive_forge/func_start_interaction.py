import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def start_interaction(self):
    """
        Print the banner and issue the first prompt.
        """
    self.text.image_create(Tk_.END, image=self.icon)
    banner_label = Tk_.Label(self.text, text=self.banner, background='#ec0fffec0', foreground='DarkGreen', anchor=Tk_.W, justify=Tk_.LEFT, font=self.settings['font'])
    self.text.window_create(Tk_.END, window=banner_label)
    self.text.insert(Tk_.END, '\n')
    self.text.mark_set('output_end', '2.0')
    try:
        home = os.environ['HOME']
    except KeyError:
        home = os.path.expanduser('~')
    desktop = os.path.join(home, 'Desktop')
    default_save_dir = desktop if os.path.exists(desktop) else home
    self.IP.magics_manager.magics['line']['cd']('-q ' + default_save_dir)
    self.interact_prompt()