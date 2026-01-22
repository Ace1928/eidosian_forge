from __future__ import annotations
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
def compute_hll_array(obj, b):
    if not 8 <= b <= 16:
        raise ValueError('b should be between 8 and 16')
    num_bits_discarded = 32 - b
    m = 1 << b
    hashes = hash_pandas_object(obj, index=False)
    if isinstance(hashes, pd.Series):
        hashes = hashes._values
    hashes = hashes.astype(np.uint32)
    j = hashes >> num_bits_discarded
    first_bit = compute_first_bit(hashes)
    df = pd.DataFrame({'j': j, 'first_bit': first_bit})
    series = df.groupby('j').max()['first_bit']
    return series.reindex(np.arange(m), fill_value=0).values.astype(np.uint8)