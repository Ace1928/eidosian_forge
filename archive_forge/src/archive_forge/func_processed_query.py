import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def processed_query(self):
    new = []
    for term in self.model.query.split():
        term = re.sub('\\.', '[^/ ]', term)
        if re.match('[A-Z]+$', term):
            new.append(BOUNDARY + WORD_OR_TAG + '/' + term + BOUNDARY)
        elif '/' in term:
            new.append(BOUNDARY + term + BOUNDARY)
        else:
            new.append(BOUNDARY + term + '/' + WORD_OR_TAG + BOUNDARY)
    return ' '.join(new)