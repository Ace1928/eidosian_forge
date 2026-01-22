from __future__ import absolute_import, print_function, division
from collections import OrderedDict
from petl.test.failonerror import assert_failonerror
from petl.test.helpers import ieq
from petl.transform.maps import fieldmap, rowmap, rowmapmany
from functools import partial
def test_recordmap():
    table = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, 'female', 17, 1.78, 74.4), (4, 'male', 21, 1.33, 45.2), (5, '-', 25, 1.65, 51.9))

    def recmapper(rec):
        transmf = {'male': 'M', 'female': 'F'}
        return [rec['id'], transmf[rec['sex']] if rec['sex'] in transmf else rec['sex'], rec['age'] * 12, rec['weight'] / rec['height'] ** 2]
    actual = rowmap(table, recmapper, header=['subject_id', 'gender', 'age_months', 'bmi'])
    expect = (('subject_id', 'gender', 'age_months', 'bmi'), (1, 'M', 16 * 12, 62.0 / 1.45 ** 2), (2, 'F', 19 * 12, 55.4 / 1.34 ** 2), (3, 'F', 17 * 12, 74.4 / 1.78 ** 2), (4, 'M', 21 * 12, 45.2 / 1.33 ** 2), (5, '-', 25 * 12, 51.9 / 1.65 ** 2))
    ieq(expect, actual)
    ieq(expect, actual)
    table2 = (('id', 'sex', 'age', 'height', 'weight'), (1, 'male', 16, 1.45, 62.0), (2, 'female', 19, 1.34, 55.4), (3, 'female', 17, 1.78, 74.4), (4, 'male', 21, 1.33, 45.2), (5, '-', 25, 1.65))
    expect = (('subject_id', 'gender', 'age_months', 'bmi'), (1, 'M', 16 * 12, 62.0 / 1.45 ** 2), (2, 'F', 19 * 12, 55.4 / 1.34 ** 2), (3, 'F', 17 * 12, 74.4 / 1.78 ** 2), (4, 'M', 21 * 12, 45.2 / 1.33 ** 2))
    actual = rowmap(table2, recmapper, header=['subject_id', 'gender', 'age_months', 'bmi'])
    ieq(expect, actual)