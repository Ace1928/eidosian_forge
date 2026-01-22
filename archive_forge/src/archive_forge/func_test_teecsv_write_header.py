from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teecsv_write_header():
    t1 = (('foo', 'bar'), ('a', '2'), ('b', '1'), ('c', '3'))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).convertnumbers().teecsv(f1.name, write_header=False, encoding='ascii').selectgt('bar', 1).tocsv(f2.name, encoding='ascii')
    ieq(t1[1:], etl.fromcsv(f1.name, encoding='ascii'))
    ieq(etl.wrap(t1).convertnumbers().selectgt('bar', 1), etl.fromcsv(f2.name, encoding='ascii').convertnumbers())