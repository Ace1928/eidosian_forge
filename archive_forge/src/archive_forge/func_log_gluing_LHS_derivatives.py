from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
def log_gluing_LHS_derivatives(self, shapes):
    """
        Compute the Jacobian of the vector-valued function f
        described in the above log_gluing_LHSs::

            sage: from snappy import Manifold
            sage: M = Manifold("m019")
            sage: shapes = M.tetrahedra_shapes('rect', bits_prec = 80)
            sage: C = KrawczykShapesEngine(M, shapes, bits_prec = 80)
            sage: RIF = RealIntervalField(80)
            sage: CIF = ComplexIntervalField(80)
            sage: shape1 = CIF(RIF(0.78055,0.78056), RIF(0.9144, 0.9145))
            sage: shape2 = CIF(RIF(0.46002,0.46003), RIF(0.6326, 0.6327))
            sage: shapes = [shape1, shape1, shape2]
            sage: C.log_gluing_LHS_derivatives(shapes) # doctest: +NUMERIC3
            [  0.292? - 1.6666?*I   0.292? - 1.6666?*I   0.752? - 1.0340?*I]
            [ 0.5400? - 0.6327?*I  0.5400? - 0.6327?*I  -1.561? - 1.8290?*I]
            [ 0.5400? - 0.6327?*I -0.5400? + 0.6327?*I                    0]

        """
    BaseField = shapes[0].parent()
    zero = BaseField(0)
    one = BaseField(1)
    shape_inverses = [one / shape for shape in shapes]
    one_minus_shape_inverses = [one / (one - shape) for shape in shapes]
    gluing_LHS_derivatives = []
    for A, B, c in self.equations:
        row = []
        for a, b, shape_inverse, one_minus_shape_inverse in zip(A, B, shape_inverses, one_minus_shape_inverses):
            derivative = zero
            if not a == 0:
                derivative = BaseField(int(a)) * shape_inverse
            if not b == 0:
                derivative -= BaseField(int(b)) * one_minus_shape_inverse
            row.append(derivative)
        gluing_LHS_derivatives.append(row)
    return matrix(BaseField, gluing_LHS_derivatives)