from ..constructor import construct_domain
from sympy.polys.domains import Domain, ZZ
def to_domain(self, domain):
    element = domain.convert_from(self.element, self.domain)
    return self.new(element, domain)