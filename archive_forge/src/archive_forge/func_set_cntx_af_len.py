import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def set_cntx_af_len(self, **kwargs):
    self._char_after = self._cntx_af_len.get()