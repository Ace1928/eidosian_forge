import functools
import re
import nltk.tree
def top_level_pred(n, m=macro_dict, l=None):
    label_dict = {}
    return any((predicate(n, m, label_dict) for predicate in tgrep_exprs))