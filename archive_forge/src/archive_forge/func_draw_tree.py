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
def draw_tree(self, edge=None):
    if edge is None and self._treetoks_edge is None:
        return
    if edge is None:
        edge = self._treetoks_edge
    if self._treetoks_edge != edge:
        self._treetoks = [t for t in self._chart.trees(edge) if isinstance(t, Tree)]
        self._treetoks_edge = edge
        self._treetoks_index = 0
    if len(self._treetoks) == 0:
        return
    for tag in self._tree_tags:
        self._tree_canvas.delete(tag)
    tree = self._treetoks[self._treetoks_index]
    self._draw_treetok(tree, edge.start())
    self._draw_treecycle()
    w = self._chart.num_leaves() * self._unitsize + 2 * ChartView._MARGIN
    h = tree.height() * (ChartView._TREE_LEVEL_SIZE + self._text_height)
    self._tree_canvas['scrollregion'] = (0, 0, w, h)