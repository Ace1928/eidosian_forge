from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq
import petl as etl
def test_teetext():
    t1 = (('foo', 'bar'), ('a', 2), ('b', 1), ('c', 3))
    f1 = NamedTemporaryFile(delete=False)
    f2 = NamedTemporaryFile(delete=False)
    prologue = 'foo,bar\n'
    template = '{foo},{bar}\n'
    epilogue = 'd,4'
    etl.wrap(t1).teetext(f1.name, template=template, prologue=prologue, epilogue=epilogue).selectgt('bar', 1).topickle(f2.name)
    ieq(t1 + (('d', 4),), etl.fromcsv(f1.name).convertnumbers())
    ieq(etl.wrap(t1).selectgt('bar', 1), etl.frompickle(f2.name))