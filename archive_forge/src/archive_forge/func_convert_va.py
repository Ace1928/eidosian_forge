import math
import warnings
import matplotlib.dates
def convert_va(mpl_va):
    """Convert mpl vertical alignment word to equivalent HTML word.

    Text alignment specifiers from mpl differ very slightly from those used
    in HTML. See the VA_MAP for more details.

    Positional arguments:
    mpl_va -- vertical mpl text alignment spec.

    """
    if mpl_va in VA_MAP:
        return VA_MAP[mpl_va]
    else:
        return None