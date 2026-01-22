import sys
from unittest import TestCase
import simplejson as json
def test_array_decoder_issue46(self):
    for doc in [u'[,]', '[,]']:
        try:
            json.loads(doc)
        except json.JSONDecodeError:
            e = sys.exc_info()[1]
            self.assertEqual(e.pos, 1)
            self.assertEqual(e.lineno, 1)
            self.assertEqual(e.colno, 2)
        except Exception:
            e = sys.exc_info()[1]
            self.fail('Unexpected exception raised %r %s' % (e, e))
        else:
            self.fail("Unexpected success parsing '[,]'")