from .cartan_type import CartanType
from mpmath import fac
from sympy.core.backend import Matrix, eye, Rational, igcd
from sympy.core.basic import Atom
def element_order(self, weylelt):
    """
        This method returns the order of a given Weyl group element, which should
        be specified by the user in the form of products of the generating
        reflections, i.e. of the form r1*r2 etc.

        For types A-F, this method current works by taking the matrix form of
        the specified element, and then finding what power of the matrix is the
        identity.  It then returns this power.

        Examples
        ========

        >>> from sympy.liealgebras.weyl_group import WeylGroup
        >>> b = WeylGroup("B4")
        >>> b.element_order('r1*r4*r2')
        4
        """
    n = self.cartan_type.rank()
    if self.cartan_type.series == 'A':
        a = self.matrix_form(weylelt)
        order = 1
        while a != eye(n + 1):
            a *= self.matrix_form(weylelt)
            order += 1
        return order
    if self.cartan_type.series == 'D':
        a = self.matrix_form(weylelt)
        order = 1
        while a != eye(n):
            a *= self.matrix_form(weylelt)
            order += 1
        return order
    if self.cartan_type.series == 'E':
        a = self.matrix_form(weylelt)
        order = 1
        while a != eye(8):
            a *= self.matrix_form(weylelt)
            order += 1
        return order
    if self.cartan_type.series == 'G':
        elts = list(weylelt)
        reflections = elts[1::3]
        m = self.delete_doubles(reflections)
        while self.delete_doubles(m) != m:
            m = self.delete_doubles(m)
            reflections = m
        if len(reflections) % 2 == 1:
            return 2
        elif len(reflections) == 0:
            return 1
        else:
            if len(reflections) == 1:
                return 2
            else:
                m = len(reflections) // 2
                lcm = 6 * m / igcd(m, 6)
            order = lcm / m
            return order
    if self.cartan_type.series == 'F':
        a = self.matrix_form(weylelt)
        order = 1
        while a != eye(4):
            a *= self.matrix_form(weylelt)
            order += 1
        return order
    if self.cartan_type.series in ('B', 'C'):
        a = self.matrix_form(weylelt)
        order = 1
        while a != eye(n):
            a *= self.matrix_form(weylelt)
            order += 1
        return order