from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import gzip
import os
import io
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_totext_gz():
    table = (('foo', 'bar'), ('a', 1), ('b', 2), ('c', 2))
    f = NamedTemporaryFile(delete=False)
    f.close()
    fn = f.name + '.gz'
    os.rename(f.name, fn)
    prologue = "{| class='wikitable'\n|-\n! foo\n! bar\n"
    template = '|-\n| {foo}\n| {bar}\n'
    epilogue = '|}\n'
    totext(table, fn, encoding='ascii', template=template, prologue=prologue, epilogue=epilogue)
    o = gzip.open(fn, 'rb')
    try:
        actual = o.read()
        expect = b"{| class='wikitable'\n|-\n! foo\n! bar\n|-\n| a\n| 1\n|-\n| b\n| 2\n|-\n| c\n| 2\n|}\n"
        eq_(expect, actual)
    finally:
        o.close()