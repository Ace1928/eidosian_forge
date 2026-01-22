from sympy.polys.rings import vring
from sympy.polys.fields import vfield
from sympy.polys.domains import QQ
def test_vfield():
    ns = {'vfield': vfield, 'QQ': QQ}
    exec('F = vfield("f", QQ)', ns)
    exec('assert f == F.gens[0]', ns)
    exec('F = vfield("fb fbb fcc fzz _fx", QQ)', ns)
    exec('assert fb == F.gens[0]', ns)
    exec('assert fbb == F.gens[1]', ns)
    exec('assert fcc == F.gens[2]', ns)
    exec('assert fzz == F.gens[3]', ns)
    exec('assert _fx == F.gens[4]', ns)
    exec('F = vfield(["fd", "fe", "ffg"], QQ)', ns)
    exec('assert fd == F.gens[0]', ns)
    exec('assert fe == F.gens[1]', ns)
    exec('assert ffg == F.gens[2]', ns)