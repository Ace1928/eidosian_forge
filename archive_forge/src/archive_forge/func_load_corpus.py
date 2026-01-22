import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def load_corpus(self, name):
    self.selected_corpus = name
    self.tagged_sents = []
    runner_thread = self.LoadCorpus(name, self)
    runner_thread.start()