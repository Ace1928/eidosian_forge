from .polynomial import Polynomial
from .fieldExtensions import my_rnfequation
from ..pari import pari
def update_assignments_and_merge(assignments, d):
    variables = sorted(set(sum([poly.variables() for poly in assignments.values()], [])))
    monomial_to_value = {(): pari(1)}
    for var in variables:
        max_degree = max([poly.degree(var) for poly in assignments.values()])
        old_keys = list(monomial_to_value.keys())
        v = d[var]
        power_of_v = pari(1)
        for degree in range(1, max_degree + 1):
            power_of_v = power_of_v * v
            for old_key in old_keys:
                old_value = monomial_to_value[old_key]
                new_key = old_key + ((var, degree),)
                new_value = old_value * power_of_v
                monomial_to_value[new_key] = new_value

    def eval_monomial(monomial):
        return pari(monomial.get_coefficient()) * monomial_to_value[monomial.get_vars()]

    def substitute(poly):
        return sum([eval_monomial(m) for m in poly.get_monomials()], pari(0))
    new_assignments = dict([(key, substitute(poly)) for key, poly in assignments.items()])
    new_assignments.update(d)
    new_assignments['1'] = pari(1)
    return new_assignments