import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def set_result_size(self, **kwargs):
    self.model.result_count = self._result_size.get()