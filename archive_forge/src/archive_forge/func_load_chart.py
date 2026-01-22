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
def load_chart(self, *args):
    """Load a chart from a pickle file"""
    filename = askopenfilename(filetypes=self.CHART_FILE_TYPES, defaultextension='.pickle')
    if not filename:
        return
    try:
        with open(filename, 'rb') as infile:
            chart = pickle.load(infile)
        self._chart = chart
        self._cv.update(chart)
        if self._matrix:
            self._matrix.set_chart(chart)
        if self._matrix:
            self._matrix.deselect_cell()
        if self._results:
            self._results.set_chart(chart)
        self._cp.set_chart(chart)
    except Exception as e:
        raise
        showerror('Error Loading Chart', 'Unable to open file: %r' % filename)