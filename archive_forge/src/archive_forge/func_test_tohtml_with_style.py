from __future__ import absolute_import, print_function, division
from tempfile import NamedTemporaryFile
import io
from petl.test.helpers import eq_
from petl.io.html import tohtml
def test_tohtml_with_style():
    table = (('foo', 'bar'), ('a', 1))
    f = NamedTemporaryFile(delete=False)
    tohtml(table, f.name, encoding='ascii', lineterminator='\n', tr_style='text-align: right', td_styles='text-align: center')
    with io.open(f.name, mode='rt', encoding='ascii', newline='') as o:
        actual = o.read()
        expect = u"<table class='petl'>\n<thead>\n<tr>\n<th>foo</th>\n<th>bar</th>\n</tr>\n</thead>\n<tbody>\n<tr style='text-align: right'>\n<td style='text-align: center'>a</td>\n<td style='text-align: center'>1</td>\n</tr>\n</tbody>\n</table>\n"
        eq_(expect, actual)