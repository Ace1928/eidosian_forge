import functools
from . import overrides
from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
from ._multiarray_umath import (
from_dlpack.__module__ = 'numpy'
frombuffer.__module__ = 'numpy'
fromfile.__module__ = 'numpy'
fromiter.__module__ = 'numpy'
frompyfunc.__module__ = 'numpy'
fromstring.__module__ = 'numpy'

    datetime_as_string(arr, unit=None, timezone='naive', casting='same_kind')

    Convert an array of datetimes into an array of strings.

    Parameters
    ----------
    arr : array_like of datetime64
        The array of UTC timestamps to format.
    unit : str
        One of None, 'auto', or a :ref:`datetime unit <arrays.dtypes.dateunits>`.
    timezone : {'naive', 'UTC', 'local'} or tzinfo
        Timezone information to use when displaying the datetime. If 'UTC', end
        with a Z to indicate UTC time. If 'local', convert to the local timezone
        first, and suffix with a +-#### timezone offset. If a tzinfo object,
        then do as with 'local', but use the specified timezone.
    casting : {'no', 'equiv', 'safe', 'same_kind', 'unsafe'}
        Casting to allow when changing between datetime units.

    Returns
    -------
    str_arr : ndarray
        An array of strings the same shape as `arr`.

    Examples
    --------
    >>> import pytz
    >>> d = np.arange('2002-10-27T04:30', 4*60, 60, dtype='M8[m]')
    >>> d
    array(['2002-10-27T04:30', '2002-10-27T05:30', '2002-10-27T06:30',
           '2002-10-27T07:30'], dtype='datetime64[m]')

    Setting the timezone to UTC shows the same information, but with a Z suffix

    >>> np.datetime_as_string(d, timezone='UTC')
    array(['2002-10-27T04:30Z', '2002-10-27T05:30Z', '2002-10-27T06:30Z',
           '2002-10-27T07:30Z'], dtype='<U35')

    Note that we picked datetimes that cross a DST boundary. Passing in a
    ``pytz`` timezone object will print the appropriate offset

    >>> np.datetime_as_string(d, timezone=pytz.timezone('US/Eastern'))
    array(['2002-10-27T00:30-0400', '2002-10-27T01:30-0400',
           '2002-10-27T01:30-0500', '2002-10-27T02:30-0500'], dtype='<U39')

    Passing in a unit will change the precision

    >>> np.datetime_as_string(d, unit='h')
    array(['2002-10-27T04', '2002-10-27T05', '2002-10-27T06', '2002-10-27T07'],
          dtype='<U32')
    >>> np.datetime_as_string(d, unit='s')
    array(['2002-10-27T04:30:00', '2002-10-27T05:30:00', '2002-10-27T06:30:00',
           '2002-10-27T07:30:00'], dtype='<U38')

    'casting' can be used to specify whether precision can be changed

    >>> np.datetime_as_string(d, unit='h', casting='safe')
    Traceback (most recent call last):
        ...
    TypeError: Cannot create a datetime string as units 'h' from a NumPy
    datetime with units 'm' according to the rule 'safe'
    