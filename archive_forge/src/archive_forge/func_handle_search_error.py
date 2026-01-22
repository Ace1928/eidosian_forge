import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def handle_search_error(self, event):
    self.status['text'] = 'Error in query ' + self.model.query
    self.unfreeze_editable()