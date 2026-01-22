from itertools import product, chain
from ..sage_helper import _within_sage
from ..math_basics import prod
from ..pari import pari
from .fundamental_polyhedron import *
def lift_to_SL2C(self):
    MatrixRepresentation.lift_to_SL2C(self)
    phi = MapToFreeAbelianization(self)
    meridian = self.peripheral_curves()[0][0]
    meridian_trace = self(meridian).trace()
    if phi.rank == 1 and phi(meridian) % 2 != 0 and (meridian_trace < 0):

        def twist(g, gen_image):
            return gen_image if phi(g)[0] % 2 == 0 else -gen_image
        self._matrices = [twist(g, M) for g, M in zip(self._gens, self._matrices)]
        self._build_hom_dict()
        assert self.is_nonprojective_representation()
        assert self(meridian).trace() > 0