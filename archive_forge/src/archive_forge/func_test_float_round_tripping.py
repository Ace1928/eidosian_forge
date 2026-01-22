from itertools import count
import platform
import numpy as np
import h5py
from .common import ut, TestCase
def test_float_round_tripping(self):
    dtypes = set((f for f in np.sctypeDict.values() if np.issubdtype(f, np.floating) or np.issubdtype(f, np.complexfloating)))
    unsupported_types = []
    if platform.machine() in UNSUPPORTED_LONG_DOUBLE:
        for x in UNSUPPORTED_LONG_DOUBLE_TYPES:
            if hasattr(np, x):
                unsupported_types.append(getattr(np, x))
    dtype_dset_map = {str(j): d for j, d in enumerate(dtypes) if d not in unsupported_types}
    fname = self.mktemp()
    with h5py.File(fname, 'w') as f:
        for n, d in dtype_dset_map.items():
            data = np.zeros(10, dtype=d)
            data[...] = np.arange(10)
            f.create_dataset(n, data=data)
    with h5py.File(fname, 'r') as f:
        for n, d in dtype_dset_map.items():
            ldata = f[n][:]
            self.assertEqual(ldata.dtype, d)