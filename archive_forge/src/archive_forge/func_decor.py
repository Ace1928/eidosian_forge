from numpy.testing import dec
from nibabel.data import DataError
def decor(f):
    for label in labels:
        setattr(f, label, True)
    return f