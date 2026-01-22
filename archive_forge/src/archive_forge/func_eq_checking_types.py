from collections import OrderedDict, deque
from datetime import date, time, datetime
from decimal import Decimal
from fractions import Fraction
import ast
import enum
import typing
def eq_checking_types(a, b):
    return type(a) is type(b) and a == b