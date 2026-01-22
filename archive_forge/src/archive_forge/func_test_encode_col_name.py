import sys
import timeit
from random import choice, randint, uniform
import pandas
import pytz
from modin.experimental.core.execution.native.implementations.hdk_on_native.dataframe.utils import (
def test_encode_col_name():

    def test(name):
        encoded = ColNameCodec.encode(name)
        assert ColNameCodec.decode(encoded) == name
    test('')
    test(None)
    test(('', ''))
    for i in range(0, 1000):
        test(randint(-sys.maxsize, sys.maxsize))
    for i in range(0, 1000):
        test(uniform(-sys.maxsize, sys.maxsize))
    for i in range(0, 1000):
        test(rnd_unicode(randint(0, 100)))
    for i in range(0, 1000):
        test((rnd_unicode(randint(0, 100)), rnd_unicode(randint(0, 100))))
    for i in range(0, 1000):
        tz = choice(pytz.all_timezones)
        test(pandas.Timestamp(randint(0, 4294967295), unit='s', tz=tz))