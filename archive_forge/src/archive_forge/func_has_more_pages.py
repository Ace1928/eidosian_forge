import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def has_more_pages(self, page):
    if self.results == [] or self.results[0] == []:
        return False
    if self.last_page is None:
        return True
    return page < self.last_page