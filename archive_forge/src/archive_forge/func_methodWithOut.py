from .. import units as pq
from .common import TestCase
import numpy as np
def methodWithOut(self, name, result, q=None, *args, **kw):
    import numpy as np
    from .. import Quantity
    if q is None:
        q = self.q
    self.assertQuantityEqual(getattr(q.copy(), name)(*args, **kw), result)
    if isinstance(result, Quantity):
        out = Quantity(np.empty_like(result.magnitude), pq.s, copy=False)
    else:
        out = np.empty_like(result)
    ret = getattr(q.copy(), name)(*args, out=out, **kw)
    self.assertQuantityEqual(ret, result)
    self.assertEqual(id(ret), id(out))
    if isinstance(result, Quantity):
        self.assertEqual(ret.units, result.units)
    else:
        self.assertEqual(getattr(ret, 'units', pq.dimensionless), pq.dimensionless)