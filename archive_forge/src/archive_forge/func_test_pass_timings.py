import unittest
from numba.tests.support import captured_stdout, override_config
def test_pass_timings(self):
    with override_config('LLVM_PASS_TIMINGS', True):
        with captured_stdout() as stdout:
            import numba

            @numba.njit
            def foo(n):
                c = 0
                for i in range(n):
                    for j in range(i):
                        c += j
                return c
            foo(10)
            md = foo.get_metadata(foo.signatures[0])
            print(md['llvm_pass_timings'])
        self.assertIn('Finalize object', stdout.getvalue())