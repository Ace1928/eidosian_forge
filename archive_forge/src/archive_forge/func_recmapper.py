from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def recmapper(rec):
    transmf = {'male': 'M', 'female': 'F'}
    return [rec['id'], transmf[rec['sex']] if rec['sex'] in transmf else rec['sex'], rec['age'] * 12, rec['weight'] / rec['height'] ** 2]