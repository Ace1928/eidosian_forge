import random
from mpmath import *
from mpmath.libmp import *
def test_custom_class():

    class mympf:

        @property
        def _mpf_(self):
            return mpf(3.5)._mpf_

    class mympc:

        @property
        def _mpc_(self):
            return (mpf(3.5)._mpf_, mpf(2.5)._mpf_)
    assert mpf(2) + mympf() == 5.5
    assert mympf() + mpf(2) == 5.5
    assert mpf(mympf()) == 3.5
    assert mympc() + mpc(2) == mpc(5.5, 2.5)
    assert mpc(2) + mympc() == mpc(5.5, 2.5)
    assert mpc(mympc()) == 3.5 + 2.5j