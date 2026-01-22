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
def save_grammar(self, *args):
    filename = asksaveasfilename(filetypes=self.GRAMMAR_FILE_TYPES, defaultextension='.cfg')
    if not filename:
        return
    try:
        if filename.endswith('.pickle'):
            with open(filename, 'wb') as outfile:
                pickle.dump((self._chart, self._tokens), outfile)
        else:
            with open(filename, 'w') as outfile:
                prods = self._grammar.productions()
                start = [p for p in prods if p.lhs() == self._grammar.start()]
                rest = [p for p in prods if p.lhs() != self._grammar.start()]
                for prod in start:
                    outfile.write('%s\n' % prod)
                for prod in rest:
                    outfile.write('%s\n' % prod)
    except Exception as e:
        showerror('Error Saving Grammar', 'Unable to open file: %r' % filename)