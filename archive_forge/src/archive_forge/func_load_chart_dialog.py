import os.path
import pickle
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from tkinter.messagebox import showerror, showinfo
from nltk.draw import CFGEditor, TreeSegmentWidget, tree_to_treesegment
from nltk.draw.util import (
from nltk.grammar import CFG, Nonterminal
from nltk.parse.chart import (
from nltk.tree import Tree
from nltk.util import in_idle
def load_chart_dialog(self, *args):
    filename = askopenfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
    if not filename:
        return
    try:
        self.load_chart(filename)
    except Exception as e:
        showerror('Error Loading Chart', f'Unable to open file: {filename!r}\n{e}')