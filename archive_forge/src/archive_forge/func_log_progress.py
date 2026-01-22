from __future__ import absolute_import, print_function, division
import abc
import logging
import sys
import time
from petl.compat import PY3
from petl.util.base import Table
from petl.util.statistics import onlinestats
def log_progress(table, batchsize=1000, prefix='', logger=None, level=logging.INFO):
    """
    Report progress on rows passing through to a python logger. If logger is
    none, a new logger will be created that, by default, streams to stdout. E.g.::

        >>> import petl as etl
        >>> table = etl.dummytable(100000)
        >>> table.log_progress(10000).tocsv('example.csv')  # doctest: +SKIP
        10000 rows in 0.13s (78363 row/s); batch in 0.13s (78363 row/s)
        20000 rows in 0.22s (91679 row/s); batch in 0.09s (110448 row/s)
        30000 rows in 0.31s (96573 row/s); batch in 0.09s (108114 row/s)
        40000 rows in 0.40s (99535 row/s); batch in 0.09s (109625 row/s)
        50000 rows in 0.49s (101396 row/s); batch in 0.09s (109591 row/s)
        60000 rows in 0.59s (102245 row/s); batch in 0.09s (106709 row/s)
        70000 rows in 0.68s (103221 row/s); batch in 0.09s (109498 row/s)
        80000 rows in 0.77s (103810 row/s); batch in 0.09s (108126 row/s)
        90000 rows in 0.90s (99465 row/s); batch in 0.13s (74516 row/s)
        100000 rows in 1.02s (98409 row/s); batch in 0.11s (89821 row/s)
        100000 rows in 1.02s (98402 row/s); batches in 0.10 +/- 0.02s [0.09-0.13] (100481 +/- 13340 rows/s [74516-110448])

    See also :func:`petl.util.timing.clock`.

    """
    return LoggingProgressView(table, batchsize, prefix, logger, level=level)