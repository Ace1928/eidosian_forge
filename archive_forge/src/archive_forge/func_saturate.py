from sympy.polys.polyerrors import CoercionFailed
from sympy.polys.polyutils import IntegerPowerable
def saturate(self, J):
    """
        Compute the ideal saturation of ``self`` by ``J``.

        That is, if ``self`` is the ideal `I`, compute the set
        `I : J^\\infty = \\{x \\in R | xJ^n \\subset I \\text{ for some } n\\}`.
        """
    raise NotImplementedError