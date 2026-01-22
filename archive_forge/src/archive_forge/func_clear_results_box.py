import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def clear_results_box(self):
    self.results_box['state'] = 'normal'
    self.results_box.delete('1.0', END)
    self.results_box['state'] = 'disabled'