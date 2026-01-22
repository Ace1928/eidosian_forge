import time
from mpmath import *
def test_hp():
    for dps in precs:
        mp.dps = dps + 8
        aa = mpf(a)
        bb = mpf(b)
        a1000 = 1000 * mpf(a)
        abi = mpc(aa, bb)
        mp.dps = dps
        assert (sqrt(3) + pi / 2).ae(aa)
        assert (e + 1 / euler ** 2).ae(bb)
        assert sqrt(aa).ae(mpf(sqrt_a))
        assert sqrt(abi).ae(mpc(sqrt_abi_real, sqrt_abi_imag))
        assert log(aa).ae(mpf(log_a))
        assert log(abi).ae(mpc(log_abi_real, log_abi_imag))
        assert exp(aa).ae(mpf(exp_a))
        assert exp(abi).ae(mpc(exp_abi_real, exp_abi_imag))
        assert (aa ** bb).ae(mpf(pow_a_b))
        assert (aa ** abi).ae(mpc(pow_a_abi_real, pow_a_abi_imag))
        assert (abi ** abi).ae(mpc(pow_abi_abi_real, pow_abi_abi_imag))
        assert sin(a).ae(mpf(sin_a))
        assert sin(a1000).ae(mpf(sin_1000a))
        assert sin(abi).ae(mpc(sin_abi_real, sin_abi_imag))
        assert cos(a).ae(mpf(cos_a))
        assert cos(a1000).ae(mpf(cos_1000a))
        assert tan(a).ae(mpf(tan_a))
        assert tan(abi).ae(mpc(tan_abi_real, tan_abi_imag))
        assert tan(abi).real.ae(mpf(tan_abi_real), abs_eps=1e-205)
        assert tan(abi).imag.ae(mpf(tan_abi_imag), abs_eps=1e-205)
    mp.dps = 460
    assert str(log(3))[-20:] == '02166121184001409826'
    mp.dps = 15