import csv
import difflib
from io import StringIO
from lxml import etree
from .jsonutil import JsonTable, get_column, get_where, get_selection
from .errors import is_xnat_error, catch_error
from .errors import ProgrammingError, NotSupportedError
from .errors import DataError, DatabaseError
def rpn_contraints(rpn_exp):
    left = []
    right = []
    triple = []
    for i, t in enumerate(rpn_exp.split()):
        if t in ['AND', 'OR']:
            if 'AND' in right or ('OR' in right and left == []):
                try:
                    operator = right.pop(right.index('AND'))
                except Exception:
                    operator = right.pop(right.index('OR'))
                left = [right[0]]
                left.append(right[1:] + [t])
                left.append(operator)
                right = []
            elif right != []:
                right.append(t)
                if left != []:
                    left.append(right)
                else:
                    left = right[:]
                    right = []
            elif right == [] and left != []:
                left = [left]
                left.append(t)
                right = left[:]
                left = []
            else:
                raise ProgrammingError('in expression %s' % rpn_exp)
        else:
            triple.append(t)
            if len(triple) == 3:
                right.append(tuple(triple))
                triple = []
    return left if left != [] else right