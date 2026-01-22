import os
import sys
import re
from urllib.request import pathname2url
from IPython.utils import io
from IPython.core.autocall import IPyAutocall
import snappy
from .gui import *
from tkinter.messagebox import askyesno
def interact_prompt(self):
    """
        Print an input prompt or a continuation prompt.  For an input
        prompt set the output_end mark at the end of the prompt.
        """
    if self.showing_traceback:
        self.showing_traceback = False
        return
    try:
        if self.IP.more or self.editing_hist:
            self.write_continuation_prompt()
        else:
            if int(self.text.index('output_end').split('.')[1]) != 0:
                self.write('\n\n', mark='output_end')
            prompt_tokens = self._input_prompt()
            for style, text in prompt_tokens:
                self.write(text, style, mark='output_end')
    except:
        self.IP.showtraceback()