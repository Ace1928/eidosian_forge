from __future__ import absolute_import, print_function, division
import io
from tempfile import NamedTemporaryFile
from petl.test.helpers import ieq, eq_
from petl.io.text import fromtext, totext
def test_fromtext():
    data = u'name,id\nԱրամ Խաչատրյան,1\nJohann Strauß,2\nВагиф Сәмәдоғлу,3\n章子怡,4\n'
    fn = NamedTemporaryFile().name
    f = io.open(fn, encoding='utf-8', mode='wt')
    f.write(data)
    f.close()
    actual = fromtext(fn, encoding='utf-8')
    expect = ((u'lines',), (u'name,id',), (u'Արամ Խաչատրյան,1',), (u'Johann Strauß,2',), (u'Вагиф Сәмәдоғлу,3',), (u'章子怡,4',))
    ieq(expect, actual)
    ieq(expect, actual)