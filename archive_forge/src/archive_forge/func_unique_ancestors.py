import functools
import re
import nltk.tree
def unique_ancestors(node):
    """
    Returns the list of all nodes dominating the given node, where
    there is only a single path of descent.
    """
    results = []
    try:
        current = node.parent()
    except AttributeError:
        return results
    while current and len(current) == 1:
        results.append(current)
        current = current.parent()
    return results