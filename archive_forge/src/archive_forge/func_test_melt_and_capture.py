from __future__ import absolute_import, print_function, division
from datetime import datetime
import pytest
from petl.errors import FieldSelectionError
from petl.test.helpers import ieq
from petl.transform.reshape import melt, recast, transpose, pivot, flatten, \
from petl.transform.regex import split, capture
def test_melt_and_capture():
    table = (('id', 'parad0', 'parad1', 'parad2'), ('1', '12', '34', '56'), ('2', '23', '45', '67'))
    expectation = (('id', 'parasitaemia', 'day'), ('1', '12', '0'), ('1', '34', '1'), ('1', '56', '2'), ('2', '23', '0'), ('2', '45', '1'), ('2', '67', '2'))
    step1 = melt(table, key='id', valuefield='parasitaemia')
    step2 = capture(step1, 'variable', 'parad(\\d+)', ('day',))
    ieq(expectation, step2)