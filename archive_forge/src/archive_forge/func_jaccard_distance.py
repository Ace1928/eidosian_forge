import operator
import warnings
def jaccard_distance(label1, label2):
    """Distance metric comparing set-similarity."""
    return (len(label1.union(label2)) - len(label1.intersection(label2))) / len(label1.union(label2))