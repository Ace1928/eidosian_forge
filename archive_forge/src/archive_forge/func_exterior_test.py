import sys
from . import links, tangles
def exterior_test():
    try:
        import snappy
    except ImportError:
        print('SnapPy not installed, skipping link exterior test.')
    print(figure8().exterior().volume(), whitehead().exterior().volume())
    C, Id = (RationalTangle(1), IdentityBraid(1))
    x = C | Id
    y = Id | -C
    print((x * y * x * y).denominator_closure().exterior().volume())
    for name, K in some_knots():
        M0, M1 = (K.exterior(), snappy.Manifold(K.DT_code(True)))
        N0 = snappy.LinkExteriors.identify(M0)
        N1 = snappy.LinkExteriors.identify(M1)
        assert N0.name() == name and N1.name() == name
    print('Checked two different ways of building 167 hyperbolic knots.')