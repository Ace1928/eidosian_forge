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
def print_selection(self, *e):
    if self._root is None:
        return
    if self._selection is None:
        showerror('Print Error', 'No tree selected')
    else:
        c = self._cframe.canvas()
        for widget in self._treewidgets:
            if widget is not self._selection:
                self._cframe.destroy_widget(widget)
        c.delete(self._selectbox)
        x1, y1, x2, y2 = self._selection.bbox()
        self._selection.move(10 - x1, 10 - y1)
        c['scrollregion'] = f'0 0 {x2 - x1 + 20} {y2 - y1 + 20}'
        self._cframe.print_to_file()
        self._treewidgets = [self._selection]
        self.clear()
        self.update()