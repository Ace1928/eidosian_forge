from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_rowmapmany():
    table = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, '-', 17, 1.78, 74.4), (4, 'male', 21, 1.33))

    def rowgenerator(row):
        transmf = {'male': 'M', 'female': 'F'}
        yield [row[0], 'gender', transmf[row[1]] if row[1] in transmf else row[1]]
        yield [row[0], 'age_months', row[2] * 12]
        yield [row[0], 'bmi', row[4] / row[3] ** 2]
    actual = rowmapmany(table, rowgenerator, header=['subject_id', 'variable', 'value'])
    expect = (('subject_id', 'variable', 'value'), (1, 'gender', 'M'), (1, 'age_months', 16 * 12), (1, 'bmi', 62.0 / 1.45 ** 2), (2, 'gender', 'F'), (2, 'age_months', 19 * 12), (2, 'bmi', 55.4 / 1.34 ** 2), (3, 'gender', '-'), (3, 'age_months', 17 * 12), (3, 'bmi', 74.4 / 1.78 ** 2), (4, 'gender', 'M'), (4, 'age_months', 21 * 12))
    ieq(expect, actual)
    ieq(expect, actual)