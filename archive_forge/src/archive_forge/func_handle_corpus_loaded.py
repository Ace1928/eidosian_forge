import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def handle_corpus_loaded(self, event):
    self.status['text'] = self.var.get() + ' is loaded'
    self.unfreeze_editable()
    self.clear_all()
    self.query_box.focus_set()