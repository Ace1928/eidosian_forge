import random
import re
import textwrap
import time
from tkinter import (
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter.font import Font
from nltk.chunk import ChunkScore, RegexpChunkParser
from nltk.chunk.regexp import RegexpChunkRule
from nltk.corpus import conll2000, treebank_chunk
from nltk.draw.util import ShowText
from nltk.tree import Tree
from nltk.util import in_idle
def show_trace(self, *e):
    self._showing_trace = True
    self.trace_button['state'] = 'disabled'
    self.devset_button['state'] = 'normal'
    self.devsetbox['state'] = 'normal'
    self.devsetbox.delete('1.0', 'end')
    self.devsetlabel['text'] = 'Development Set (%d/%d)' % (self.devset_index + 1, self._devset_size.get())
    if self.chunker is None:
        self.devsetbox.insert('1.0', 'Trace: waiting for a valid grammar.')
        self.devsetbox.tag_add('error', '1.0', 'end')
        return
    gold_tree = self.devset[self.devset_index]
    rules = self.chunker.rules()
    tagseq = '\t'
    charnum = [1]
    for wordnum, (word, pos) in enumerate(gold_tree.leaves()):
        tagseq += '%s ' % pos
        charnum.append(len(tagseq))
    self.charnum = {(i, j): charnum[j] for i in range(len(rules) + 1) for j in range(len(charnum))}
    self.linenum = {i: i * 2 + 2 for i in range(len(rules) + 1)}
    for i in range(len(rules) + 1):
        if i == 0:
            self.devsetbox.insert('end', 'Start:\n')
            self.devsetbox.tag_add('trace', 'end -2c linestart', 'end -2c')
        else:
            self.devsetbox.insert('end', 'Apply %s:\n' % rules[i - 1])
            self.devsetbox.tag_add('trace', 'end -2c linestart', 'end -2c')
        self.devsetbox.insert('end', tagseq + '\n')
        self.devsetbox.tag_add('wrapindent', 'end -2c linestart', 'end -2c')
        chunker = RegexpChunkParser(rules[:i])
        test_tree = self._chunkparse(gold_tree.leaves())
        gold_chunks = self._chunks(gold_tree)
        test_chunks = self._chunks(test_tree)
        for chunk in gold_chunks.intersection(test_chunks):
            self._color_chunk(i, chunk, 'true-pos')
        for chunk in gold_chunks - test_chunks:
            self._color_chunk(i, chunk, 'false-neg')
        for chunk in test_chunks - gold_chunks:
            self._color_chunk(i, chunk, 'false-pos')
    self.devsetbox.insert('end', 'Finished.\n')
    self.devsetbox.tag_add('trace', 'end -2c linestart', 'end -2c')
    self.top.after(100, self.devset_xscroll.set, 0, 0.3)