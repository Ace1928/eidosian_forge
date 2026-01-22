import math
import pytest
from mpmath import *
def test_lambertw_hard():

    def check(x, y):
        y = convert(y)
        type_ok = True
        if isinstance(y, mpf):
            type_ok = isinstance(x, mpf)
        real_ok = abs(x.real - y.real) <= abs(y.real) * 8 * eps
        imag_ok = abs(x.imag - y.imag) <= abs(y.imag) * 8 * eps
        return real_ok and imag_ok
    mp.dps = 15
    assert check(lambertw(1e-10), 9.999999999e-11)
    assert check(lambertw(-1e-10), -1.0000000001e-10)
    assert check(lambertw(1e-10j), 1e-20 + 1e-10j)
    assert check(lambertw(-1e-10j), 1e-20 - 1e-10j)
    assert check(lambertw(1e-10, 1), -26.303186778379043 + 3.2650939117038282j)
    assert check(lambertw(-1e-10, 1), -26.326236166739164 + 6.526183280686333j)
    assert check(lambertw(1e-10j, 1), -26.312931726911422 + 4.896366881798014j)
    assert check(lambertw(-1e-10j, 1), -26.297238779529035 + 1.6328071613455766j)
    assert check(lambertw(1e-10, -1), -26.303186778379043 - 3.2650939117038282j)
    assert check(lambertw(-1e-10, -1), -26.295238819246926)
    assert check(lambertw(1e-10j, -1), -26.297238779529035 - 1.6328071613455766j)
    assert check(lambertw(-1e-10j, -1), -26.312931726911422 - 4.896366881798014j)
    add = lambda x, y: fadd(x, y, exact=True)
    sub = lambda x, y: fsub(x, y, exact=True)
    addj = lambda x, y: fadd(x, fmul(y, 1j, exact=True), exact=True)
    subj = lambda x, y: fadd(x, fmul(y, -1j, exact=True), exact=True)
    mp.dps = 1500
    a = -1 / e + 10 * eps
    d3 = mpf('1e-3')
    d10 = mpf('1e-10')
    d20 = mpf('1e-20')
    d40 = mpf('1e-40')
    d80 = mpf('1e-80')
    d300 = mpf('1e-300')
    d1000 = mpf('1e-1000')
    mp.dps = 15
    assert check(lambertw(add(a, d3)), -0.9280201500545671)
    assert check(lambertw(add(a, d10)), -0.9999766837414009)
    assert check(lambertw(add(a, d20)), -0.9999999997668356)
    assert lambertw(add(a, d40)) == -1
    assert lambertw(add(a, d80)) == -1
    assert lambertw(add(a, d300)) == -1
    assert lambertw(add(a, d1000)) == -1
    assert check(lambertw(sub(a, d3)), -0.9981901614986098 + 0.07367191188934638j)
    assert check(lambertw(sub(a, d10)), -0.9999999998187812 + 2.331643981403461e-05j)
    assert check(lambertw(sub(a, d20)), -1.0 + 2.331643981597124e-10j)
    assert check(lambertw(sub(a, d40)), -1.0 + 2.3316439815971243e-20j)
    assert check(lambertw(sub(a, d80)), -1.0 + 2.331643981597124e-40j)
    assert check(lambertw(sub(a, d300)), -1.0 + 2.3316439815971243e-150j)
    assert check(lambertw(sub(a, d1000)), mpc(-1, '2.33164398159712420336e-500'))
    assert check(lambertw(addj(a, d3)), -0.9479038748693853 + 0.05036819639190133j)
    assert check(lambertw(addj(a, d10)), -0.9999835127872944 + 1.648703148958212e-05j)
    assert check(lambertw(addj(a, d20)), -0.9999999998351279 + 1.6487212705189094e-10j)
    assert check(lambertw(addj(a, d40)), -1.0 + 1.6487212707001282e-20j)
    assert check(lambertw(addj(a, d80)), -1.0 + 1.6487212707001281e-40j)
    assert check(lambertw(addj(a, d300)), -1.0 + 1.648721270700128e-150j)
    assert check(lambertw(addj(a, d1000)), mpc(-1.0, '1.64872127070012814684865e-500'))
    assert check(lambertw(subj(a, d3)), -0.9479038748693853 - 0.05036819639190133j)
    assert check(lambertw(subj(a, d10)), -0.9999835127872944 - 1.648703148958212e-05j)
    assert check(lambertw(subj(a, d20)), -0.9999999998351279 - 1.6487212705189094e-10j)
    assert check(lambertw(subj(a, d40)), -1.0 - 1.6487212707001282e-20j)
    assert check(lambertw(subj(a, d80)), -1.0 - 1.6487212707001281e-40j)
    assert check(lambertw(subj(a, d300)), -1.0 - 1.648721270700128e-150j)
    assert check(lambertw(subj(a, d1000)), mpc(-1.0, '-1.64872127070012814684865e-500'))
    assert check(lambertw(addj(a, d3), 1), -3.0885013032199335 + 7.458676867597474j)
    assert check(lambertw(addj(a, d80), 1), -3.088843015613044 + 7.461489285654254j)
    assert check(lambertw(addj(a, d300), 1), -3.088843015613044 + 7.461489285654254j)
    assert check(lambertw(addj(a, d1000), 1), -3.088843015613044 + 7.461489285654254j)
    assert check(lambertw(subj(a, d3), 1), -1.052091418045013 + 0.05399256381254505j)
    assert check(lambertw(subj(a, d10), 1), -1.0000164872127055 + 1.648739392715925e-05j)
    assert check(lambertw(subj(a, d20), 1), -1.0000000001648721 + 1.648721270881347e-10j)
    assert check(lambertw(subj(a, d40), 1), -1.0 + 1.6487212707001282e-20j)
    assert check(lambertw(subj(a, d80), 1), -1.0 + 1.6487212707001281e-40j)
    assert check(lambertw(subj(a, d300), 1), -1.0 + 1.648721270700128e-150j)
    assert check(lambertw(subj(a, d1000), 1), mpc(-1.0, '1.64872127070012814684865e-500'))
    assert check(lambertw(add(a, d3), -1), -1.075608941186625)
    assert check(lambertw(add(a, d10), -1), -1.0000233166210366)
    assert check(lambertw(add(a, d20), -1), -1.0000000002331644)
    assert lambertw(add(a, d40), -1) == -1
    assert lambertw(add(a, d80), -1) == -1
    assert lambertw(add(a, d300), -1) == -1
    assert lambertw(add(a, d1000), -1) == -1
    assert check(lambertw(sub(a, d3), -1), -0.9981901614986098 - 0.07367191188934638j)
    assert check(lambertw(sub(a, d10), -1), -0.9999999998187812 - 2.331643981403461e-05j)
    assert check(lambertw(sub(a, d20), -1), -1.0 - 2.331643981597124e-10j)
    assert check(lambertw(sub(a, d40), -1), -1.0 - 2.3316439815971243e-20j)
    assert check(lambertw(sub(a, d80), -1), -1.0 - 2.331643981597124e-40j)
    assert check(lambertw(sub(a, d300), -1), -1.0 - 2.3316439815971243e-150j)
    assert check(lambertw(sub(a, d1000), -1), mpc(-1, '-2.33164398159712420336e-500'))
    assert check(lambertw(addj(a, d3), -1), -1.052091418045013 - 0.05399256381254505j)
    assert check(lambertw(addj(a, d10), -1), -1.0000164872127055 - 1.648739392715925e-05j)
    assert check(lambertw(addj(a, d20), -1), -1.0000000001648721 - 1.648721270881347e-10j)
    assert check(lambertw(addj(a, d40), -1), -1.0 - 1.6487212707001282e-20j)
    assert check(lambertw(addj(a, d80), -1), -1.0 - 1.6487212707001281e-40j)
    assert check(lambertw(addj(a, d300), -1), -1.0 - 1.648721270700128e-150j)
    assert check(lambertw(addj(a, d1000), -1), mpc(-1.0, '-1.64872127070012814684865e-500'))
    assert check(lambertw(subj(a, d3), -1), -3.0885013032199335 - 7.458676867597474j)
    assert check(lambertw(subj(a, d10), -1), -3.0888430155792608 - 7.461489285372969j)
    assert check(lambertw(subj(a, d20), -1), -3.088843015613044 - 7.461489285654254j)
    assert check(lambertw(subj(a, d40), -1), -3.088843015613044 - 7.461489285654254j)
    assert check(lambertw(subj(a, d80), -1), -3.088843015613044 - 7.461489285654254j)
    assert check(lambertw(subj(a, d300), -1), -3.088843015613044 - 7.461489285654254j)
    assert check(lambertw(subj(a, d1000), -1), -3.088843015613044 - 7.461489285654254j)
    mp.dps = 500
    x = -1 / e + mpf('1e-13')
    ans = '-0.99999926266961377166355784455394913638782494543377383744978844374498153493943725364881490261187530235150668593869563168276697689459394902153960200361935311512317183678882'
    mp.dps = 15
    assert lambertw(x).ae(ans)
    mp.dps = 50
    assert lambertw(x).ae(ans)
    mp.dps = 150
    assert lambertw(x).ae(ans)