from typing import Tuple, Union
import numpy as np
def float32_to_4bit_unpacked(x: Union[np.ndarray, np.dtype, float], signed: bool) -> np.ndarray:
    """Cast to 4bit via rounding and clipping (without packing).

    Args:
        x: element to be converted
        signed: boolean, whether to convert to signed int4.

    Returns:
        An ndarray with a single int4 element (sign-extended to int8/uint8)
    """
    dtype = np.int8 if signed else np.uint8
    clip_low = INT4_MIN if signed else UINT4_MIN
    clip_high = INT4_MAX if signed else UINT4_MAX
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    return np.rint(np.clip(x, clip_low, clip_high)).astype(dtype)