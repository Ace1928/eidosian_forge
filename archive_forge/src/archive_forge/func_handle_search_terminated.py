import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def handle_search_terminated(self, event):
    results = self.model.get_results()
    self.write_results(results)
    self.status['text'] = ''
    if len(results) == 0:
        self.status['text'] = 'No results found for ' + self.model.query
    else:
        self.current_page = self.model.last_requested_page
    self.unfreeze_editable()
    self.results_box.xview_moveto(self._FRACTION_LEFT_TEXT)