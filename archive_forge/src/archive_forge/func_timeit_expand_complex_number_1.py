from sympy.core import symbols, I
def timeit_expand_complex_number_1():
    ((2 + 3 * I) ** 1000).expand(complex=True)