from .polynomial import Polynomial, Monomial
from . import matrix
def new_degree(var, expo):
    if var == mod_var:
        return (var, expo - mod_degree)
    else:
        return (var, expo)