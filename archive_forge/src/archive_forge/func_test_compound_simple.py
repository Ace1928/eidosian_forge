import numpy as np
import h5py
import h5py._hl.selections as sel
import h5py._hl.selections2 as sel2
from .common import TestCase, ut
def test_compound_simple(self):
    """ Compound types with elemental subtypes """
    dt = np.dtype([('a', 'i'), ('b', 'f'), ('c', '|S10')])
    out, format = sel2.read_dtypes(dt, ())
    self.assertEqual(out, format)
    self.assertEqual(out, dt)
    out, format = sel2.read_dtypes(dt, ('a', 'b'))
    self.assertEqual(out, format)
    self.assertEqual(out, np.dtype([('a', 'i'), ('b', 'f')]))
    out, format = sel2.read_dtypes(dt, ('a',))
    self.assertEqual(out, np.dtype('i'))
    self.assertEqual(format, np.dtype([('a', 'i')]))
    with self.assertRaises(ValueError):
        out, format = sel2.read_dtypes(dt, ('j', 'k'))