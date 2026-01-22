from taskflow import test
from taskflow.utils import misc
def test_it_works_with_unicode(self):
    data = _bytes('{"foo": "фуу"}')
    self.assertEqual({'foo': u'фуу'}, misc.decode_json(data))