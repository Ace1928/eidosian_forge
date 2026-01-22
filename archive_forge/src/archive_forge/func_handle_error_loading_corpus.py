import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def handle_error_loading_corpus(self, event):
    self.status['text'] = 'Error in loading ' + self.var.get()
    self.unfreeze_editable()
    self.clear_all()
    self.freeze_editable()