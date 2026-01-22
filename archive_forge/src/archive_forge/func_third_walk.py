import numpy as np
def third_walk(tree, n):
    tree.x += n
    for c in tree.children:
        third_walk(c, n)