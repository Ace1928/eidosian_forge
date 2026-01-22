from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teecsv():
    t1 = (('foo', 'bar'), ('a', 2), ('b', 1), ('c', 3))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    etl.wrap(t1).teecsv(f1.name, encoding='ascii').selectgt('bar', 1).tocsv(f2.name, encoding='ascii')
    ieq(t1, etl.fromcsv(f1.name, encoding='ascii').convertnumbers())
    ieq(etl.wrap(t1).selectgt('bar', 1), etl.fromcsv(f2.name, encoding='ascii').convertnumbers())