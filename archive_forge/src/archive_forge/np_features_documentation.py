from functools import lru_cache
import numpy as np
Return True if ufuncs on memmap arrays always return memmap arrays

    This should be True for numpy < 1.12, False otherwise.
    