import unittest
from numba.tests.support import captured_stdout
from numba import typed
def test_ex_literal_dict_compile_time_consts(self):
    with captured_stdout():
        import numpy as np
        from numba import njit, types
        from numba.extending import overload

        def specialize(x):
            pass

        @overload(specialize)
        def ol_specialize(x):
            ld = x.literal_value
            const_expr = []
            for k, v in ld.items():
                if isinstance(v, types.Literal):
                    lv = v.literal_value
                    if lv == 'cat':
                        const_expr.append('Meow!')
                    elif lv == 'dog':
                        const_expr.append('Woof!')
                    elif isinstance(lv, int):
                        const_expr.append(k.literal_value * lv)
                else:
                    const_expr.append('Array(dim={dim}'.format(dim=v.ndim))
            const_strings = tuple(const_expr)

            def impl(x):
                return const_strings
            return impl

        @njit
        def foo():
            pets_ints_and_array = {'a': 1, 'b': 2, 'c': 'cat', 'd': 'dog', 'e': np.ones(5)}
            return specialize(pets_ints_and_array)
        result = foo()
        print(result)
    self.assertEqual(result, ('a', 'bb', 'Meow!', 'Woof!', 'Array(dim=1'))