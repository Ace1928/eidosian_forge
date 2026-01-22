import unittest
from numba.tests.support import captured_stdout
@numba.extending.overload(power)
def ov_power(x, n):
    if isinstance(n, numba.types.Literal):
        if n.literal_value == 2:
            print('square')
            return lambda x, n: x * x
        elif n.literal_value == 3:
            print('cubic')
            return lambda x, n: x * x * x
    else:
        return lambda x, n: numba.literally(n)
    print('generic')
    return lambda x, n: x ** n