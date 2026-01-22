from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import io
from petl.test.helpers import eq_
from petl.io.html import tohtml
def test_tohtml_caption():
    table = (('foo', 'bar'), ('a', 1), ('b', (1, 2)))
    f = NamedTemporaryFile(delete=False)
    tohtml(table, f.name, encoding='ascii', caption='my table', lineterminator='\n')
    with io.open(f.name, mode='rt', encoding='ascii', newline='') as o:
        actual = o.read()
        expect = u"<table class='petl'>\n<caption>my table</caption>\n<thead>\n<tr>\n<th>foo</th>\n<th>bar</th>\n</tr>\n</thead>\n<tbody>\n<tr>\n<td>a</td>\n<td style='text-align: right'>1</td>\n</tr>\n<tr>\n<td>b</td>\n<td>(1, 2)</td>\n</tr>\n</tbody>\n</table>\n"
        eq_(expect, actual)