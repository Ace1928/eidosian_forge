from numba import cuda
from numba.core.errors import TypingError
from numba.cuda.testing import unittest, CUDATestCase, skip_on_cudasim
@skip_on_cudasim('TypingError does not occur on simulator')
def test_typing_error(self):

    @cuda.jit(device=True)
    def dev_func(x):
        return floor(x)

    @cuda.jit
    def kernel_func():
        dev_func(1.5)
    with self.assertRaises(TypingError) as raises:
        kernel_func[1, 1]()
    excstr = str(raises.exception)
    self.assertIn('resolving callee type: type(CUDADispatcher', excstr)
    self.assertIn("NameError: name 'floor' is not defined", excstr)