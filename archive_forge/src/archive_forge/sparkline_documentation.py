import math
from typing import List, Union
Convert <series> to a sparkline string.

    Example:
    >>> sparkify([ 0.5, 1.2, 3.5, 7.3, 8.0, 12.5, 13.2, 15.0, 14.2, 11.8, 6.1,
    ... 1.9 ])
    u'▁▁▂▄▅▇▇██▆▄▂'

    >>> sparkify([1, 1, -2, 3, -5, 8, -13])
    u'▆▆▅▆▄█▁'

    Raises ValueError if input data cannot be converted to float.
    Raises TypeError if series is not an iterable.
    