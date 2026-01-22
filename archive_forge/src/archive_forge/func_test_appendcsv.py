from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq, eq_
from petl.io.csv import fromcsv, tocsv, appendcsv
def test_appendcsv():
    data = u'name,id\nԱրամ Խաչատրյան,1\nJohann Strauß,2\nВагиф Сәмәдоғлу,3\n章子怡,4\n'
    fn = NamedTemporaryFile().name
    uf = io.open(fn, encoding='utf-8', mode='wt')
    uf.write(data)
    uf.close()
    tbl = ((u'name', u'id'), (u'ኃይሌ ገብረሥላሴ', 5), (u'ედუარდ შევარდნაძე', 6))
    appendcsv(tbl, fn, encoding='utf-8', lineterminator='\n')
    expect = u'name,id\nԱրամ Խաչատրյան,1\nJohann Strauß,2\nВагиф Сәмәдоғлу,3\n章子怡,4\nኃይሌ ገብረሥላሴ,5\nედუარდ შევარდნაძე,6\n'
    uf = io.open(fn, encoding='utf-8', mode='rt')
    actual = uf.read()
    eq_(expect, actual)