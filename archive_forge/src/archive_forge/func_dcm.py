from sympy.core.backend import (diff, expand, sin, cos, sympify, eye, zeros,
from sympy.core.symbol import Symbol
from sympy.simplify.trigsimp import trigsimp
from sympy.physics.vector.vector import Vector, _check_vector
from sympy.utilities.misc import translate
from warnings import warn
def dcm(self, otherframe):
    """Returns the direction cosine matrix of this reference frame
        relative to the provided reference frame.

        The returned matrix can be used to express the orthogonal unit vectors
        of this frame in terms of the orthogonal unit vectors of
        ``otherframe``.

        Parameters
        ==========

        otherframe : ReferenceFrame
            The reference frame which the direction cosine matrix of this frame
            is formed relative to.

        Examples
        ========

        The following example rotates the reference frame A relative to N by a
        simple rotation and then calculates the direction cosine matrix of N
        relative to A.

        >>> from sympy import symbols, sin, cos
        >>> from sympy.physics.vector import ReferenceFrame
        >>> q1 = symbols('q1')
        >>> N = ReferenceFrame('N')
        >>> A = ReferenceFrame('A')
        >>> A.orient_axis(N, q1, N.x)
        >>> N.dcm(A)
        Matrix([
        [1,       0,        0],
        [0, cos(q1), -sin(q1)],
        [0, sin(q1),  cos(q1)]])

        The second row of the above direction cosine matrix represents the
        ``N.y`` unit vector in N expressed in A. Like so:

        >>> Ny = 0*A.x + cos(q1)*A.y - sin(q1)*A.z

        Thus, expressing ``N.y`` in A should return the same result:

        >>> N.y.express(A)
        cos(q1)*A.y - sin(q1)*A.z

        Notes
        =====

        It is important to know what form of the direction cosine matrix is
        returned. If ``B.dcm(A)`` is called, it means the "direction cosine
        matrix of B rotated relative to A". This is the matrix
        :math:`{}^B\\mathbf{C}^A` shown in the following relationship:

        .. math::

           \\begin{bmatrix}
             \\hat{\\mathbf{b}}_1 \\\\
             \\hat{\\mathbf{b}}_2 \\\\
             \\hat{\\mathbf{b}}_3
           \\end{bmatrix}
           =
           {}^B\\mathbf{C}^A
           \\begin{bmatrix}
             \\hat{\\mathbf{a}}_1 \\\\
             \\hat{\\mathbf{a}}_2 \\\\
             \\hat{\\mathbf{a}}_3
           \\end{bmatrix}.

        :math:`{}^B\\mathbf{C}^A` is the matrix that expresses the B unit
        vectors in terms of the A unit vectors.

        """
    _check_frame(otherframe)
    if otherframe in self._dcm_cache:
        return self._dcm_cache[otherframe]
    flist = self._dict_list(otherframe, 0)
    outdcm = eye(3)
    for i in range(len(flist) - 1):
        outdcm = outdcm * flist[i]._dcm_dict[flist[i + 1]]
    self._dcm_cache[otherframe] = outdcm
    otherframe._dcm_cache[self] = outdcm.T
    return outdcm