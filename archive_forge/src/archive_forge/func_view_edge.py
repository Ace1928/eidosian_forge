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
def view_edge(self, edge):
    level = None
    for i in range(len(self._edgelevels)):
        if edge in self._edgelevels[i]:
            level = i
            break
    if level is None:
        return
    y = (level + 1) * self._chart_level_size
    dy = self._text_height + 10
    self._chart_canvas.yview('moveto', 1.0)
    if self._chart_height != 0:
        self._chart_canvas.yview('moveto', (y - dy) / self._chart_height)