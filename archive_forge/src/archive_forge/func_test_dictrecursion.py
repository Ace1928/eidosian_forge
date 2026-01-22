from unittest import TestCase
import simplejson as json
def test_dictrecursion(self):
    x = {}
    x['test'] = x
    try:
        json.dumps(x)
    except ValueError:
        pass
    else:
        self.fail("didn't raise ValueError on dict recursion")
    x = {}
    y = {'a': x, 'b': x}
    json.dumps(y)