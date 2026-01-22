from parlai.core.teachers import FbDialogTeacher, MultiTaskTeacher
from .build import build
import copy
import os
def mod_labels(ys, task):
    if ys is not None:
        if task == '8':
            ys = [y.replace(',', ' ') for y in ys]
        elif task == '19':
            ys = [y.replace(',', ' ') for y in ys]
    return ys