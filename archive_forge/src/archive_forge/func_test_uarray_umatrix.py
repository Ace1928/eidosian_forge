from builtins import str
import sys
import os
def test_uarray_umatrix():
    """
        Test of the transformation of uarray(tuple,...) into
        uarray(nominal_values, std_devs). Also performs the same tests
        on umatrix().
        """
    tests = {'uarray((arange(3), std_devs))': 'uarray(arange(3), std_devs)', 'uarray(tuple_arg)': 'uarray(*tuple_arg)', 'uarray(values, std_devs)': 'uarray(values, std_devs)', 'uarray( ( arange(3),  std_devs ) ) ': 'uarray( arange(3),  std_devs) ', 'uarray(  tuple_arg )': 'uarray(*  tuple_arg)'}
    tests.update(dict(((orig.replace('uarray', 'un.uarray'), new.replace('uarray', 'un.uarray')) for orig, new in tests.items())))
    tests.update(dict(((orig + '**2', new + '**2') for orig, new in tests.items())))
    tests[' t  =  u.uarray(args)'] = ' t  =  u.uarray(*args)'
    tests.update(dict(((orig.replace('uarray', 'umatrix'), new.replace('uarray', 'umatrix')) for orig, new in tests.items())))
    check_all('uarray_umatrix', tests)