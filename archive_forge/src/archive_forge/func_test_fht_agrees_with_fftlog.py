import warnings
import numpy as np
import pytest
from scipy.fft._fftlog import fht, ifht, fhtoffset
from scipy.special import poch
from scipy.conftest import array_api_compatible
from scipy._lib._array_api import xp_assert_close
@array_api_compatible
def test_fht_agrees_with_fftlog(xp):

    def f(r, mu):
        return r ** (mu + 1) * np.exp(-r ** 2 / 2)
    r = np.logspace(-4, 4, 16)
    dln = np.log(r[1] / r[0])
    mu = 0.3
    offset = 0.0
    bias = 0.0
    a = xp.asarray(f(r, mu))
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-0.001159922613593045, +0.001625822618458832, -0.00194951828643233, +0.003789220182554077, +0.0005093959119952945, +0.02785387803618774, +0.09944952700848897, +0.4599202164586588, +0.3157462160881342, -0.0008201236844404755, -0.0007834031308271878, +0.0003931444945110708, -0.0002697710625194777, +0.000356839805023882, -0.0005554454827797206, +0.0008286331026468585]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+4.353768523152057e-05, -9.197045663594285e-06, +0.0003150140927838524, +0.0009149121960963704, +0.005808089753959363, +0.0254806525637724, +0.1339477692089897, +0.4821530509479356, +0.2659899781579785, -0.01116475278448113, +0.001791441617592385, -0.0004181810476548056, +0.0001314963536765343, -5.422057743066297e-05, +3.208681804170443e-05, -2.696849476008234e-05]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)
    bias = 0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [-7.343667355831685, +0.17102712078171, +0.1065374386206564, -0.05121739602708132, +0.0263664931926947, +0.01697209218849693, +0.1250215614723183, +0.4739583261486729, +0.2841149874912028, -0.00831276474164573, +0.001024233505508988, -0.000164490276738912, +3.30577547692627e-05, -7.786993194882709e-06, +1.962258449520547e-06, -8.97789573490925e-07]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)
    bias = -0.8
    offset = fhtoffset(dln, mu, bias=bias)
    ours = fht(a, dln, mu, offset=offset, bias=bias)
    theirs = [+8.985777068568745e-06, +4.074898209936099e-05, +0.0002123969254700955, +0.001009558244834628, +0.005131386375222176, +0.02461678673516286, +0.1235812845384476, +0.4719570096404403, +0.2893487490631317, -0.01686570611318716, +0.02231398155172505, -0.01480742256379873, +0.1692387813500801, +0.3097490354365797, +2.759360718240186, 10.52510750700458]
    theirs = xp.asarray(theirs, dtype=xp.float64)
    xp_assert_close(ours, theirs)