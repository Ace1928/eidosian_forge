import os
from itertools import chain
import nltk
from nltk.internals import Counter
from nltk.sem import drt, linearlogic
from nltk.sem.logic import (
from nltk.tag import BigramTagger, RegexpTagger, TrigramTagger, UnigramTagger
def find_label_name(self, name, node, depgraph, unique_index):
    try:
        dot = name.index('.')
        before_dot = name[:dot]
        after_dot = name[dot + 1:]
        if before_dot == 'super':
            return self.find_label_name(after_dot, depgraph.nodes[node['head']], depgraph, unique_index)
        else:
            return self.find_label_name(after_dot, self.lookup_unique(before_dot, node, depgraph), depgraph, unique_index)
    except ValueError:
        lbl = self.get_label(node)
        if name == 'f':
            return lbl
        elif name == 'v':
            return '%sv' % lbl
        elif name == 'r':
            return '%sr' % lbl
        elif name == 'super':
            return self.get_label(depgraph.nodes[node['head']])
        elif name == 'var':
            return f'{lbl.upper()}{unique_index}'
        elif name == 'a':
            return self.get_label(self.lookup_unique('conja', node, depgraph))
        elif name == 'b':
            return self.get_label(self.lookup_unique('conjb', node, depgraph))
        else:
            return self.get_label(self.lookup_unique(name, node, depgraph))