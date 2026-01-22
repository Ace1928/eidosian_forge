from .component import ZeroDimensionalComponent
from .rur import RUR
from . import matrix
from . import findLoops
from . import utilities
from ..sage_helper import _within_sage
from ..pari import Gen, pari
import re
def is_pu_2_1_representation(self, epsilon, epsilon2=None):
    """
        Returns True if the representation is also a
        PU(2,1)-representation. This uses Proposition 3.5 and the
        remark following that proposition in [FKR2013]_.

        If a condition given in that Proposition is violated, the method returns
        an object whose Boolean value is still False and that indicates which condition
        was violated. Thus, this method can still be used in ``if`` statements.

        The method tests the following complex equalities and inequalities:

        * the three complex equations given in (3.5.1) of [FKR2013]_.
        * the inequality z\\ :sub:`ijl` :math:`\\\\not=` -1.

        **Remark:** It does not check whether all z\\ :sub:`ij` * z\\ :sub:`ji` are real or
        not as these are still valid CR configurations, see the remark following
        Proposition 3.5.

        The user has to supply an epsilon: an equality/inequality is considered
        to be true if and only if the absolute value | LHS - RHS | of difference between the
        left and right hand side is less/greater than epsilon.

        The user can supply another parameter, epsilon2. If any | LHS - RHS | is in
        the interval [epsilon, epsilon2], this method fails with an exception
        as the value of | LHS - RHS | is an ambiguous interval where
        it is unclear whether inequality fails to hold because it truly does
        hold or just because of numerical noise.
        """

    def is_zero(val):
        if val.abs() < epsilon:
            return True
        if epsilon2:
            if not epsilon2 < val.abs():
                raise Exception('Ambiguous error when determining whether a condition was fulfilled or nor: %s' % val)
        return False

    def mainCondition(key_zij, key_zji, key_zkl, key_zlk):
        lhs = self[key_zij] * self[key_zji]
        rhs = (self[key_zkl] * self[key_zlk]).conj()
        if not is_zero(lhs - rhs):
            reason = '%s * %s = conjugate(%s * %s) not fulfilled' % (key_zij, key_zji, key_zkl, key_zlk)
            return NotPU21Representation(reason)
        return True

    def tripleRatioCondition(key_zji, key_zki, key_zli):
        tripleRatio = self[key_zji] * self[key_zki] * self[key_zli]
        if is_zero(tripleRatio - 1):
            reason = 'Triple ratio %s * %s * %s = 1' % (key_zji, key_zki, key_zli)
            return NotPU21Representation(reason)
        return True
    if self.N() != 3:
        raise Exception('PU(2,1)-representations only allowed for N = 3')
    if not self._is_numerical:
        raise NumericalMethodError('is_pu_2_1_representation')
    for t in range(self.num_tetrahedra()):
        m0 = mainCondition('z_1000_%d' % t, 'z_0100_%d' % t, 'z_0010_%d' % t, 'z_0001_%d' % t)
        if not m0:
            return m0
        m1 = mainCondition('zp_1000_%d' % t, 'zp_0010_%d' % t, 'zp_0100_%d' % t, 'zp_0001_%d' % t)
        if not m1:
            return m1
        m2 = mainCondition('zpp_1000_%d' % t, 'zpp_0001_%d' % t, 'zpp_0100_%d' % t, 'zpp_0010_%d' % t)
        if not m2:
            return m2
        t0 = tripleRatioCondition('z_0100_%d' % t, 'zp_0010_%d' % t, 'zpp_0001_%d' % t)
        if not t0:
            return t0
        t1 = tripleRatioCondition('z_1000_%d' % t, 'zp_0001_%d' % t, 'zpp_0010_%d' % t)
        if not t1:
            return t1
        t2 = tripleRatioCondition('z_0001_%d' % t, 'zp_1000_%d' % t, 'zpp_0100_%d' % t)
        if not t2:
            return t2
        t3 = tripleRatioCondition('z_0010_%d' % t, 'zp_0100_%d' % t, 'zpp_1000_%d' % t)
        if not t3:
            return t3
    return True