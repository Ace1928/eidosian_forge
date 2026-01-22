from sympy.liealgebras.cartan_type import Standard_Cartan
from sympy.core.backend import eye
def highest_root(self):
    """
        Returns the highest weight root for A_n
        """
    return self.basic_root(0, self.n)