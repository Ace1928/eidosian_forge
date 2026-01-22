from snappy import snap
from snappy.sage_helper import _within_sage, sage_method
def krawczyk_interval(self, shape_intervals):
    """
        Compute the interval in the Krawczyk test.

        It is given as

            K(z0, [z], f) := z0 - c * f(z0) + (Id - c * df([z])) * ([z] - z0)

        where
           - z0 is the approximate candidate solution,
           - [z] are the shape_intervals we try to verify,
           - f is the function taking the shapes to the errors of the logarithmic gluing equations
           - c is an approximate inverse of df
           - df([z]) is the derivative of f (interval-)evaluated for [z]

        Note that z0 in self.initial_shapes which are complex intervals
        containing only one value (the candidate solution given initially).

        If K is contained in [z], then we have proven that [z] contains a solution
        to the gluing equations.

        Do several Krawczyk operations to get a better solution::

            sage: M = Manifold("m019")
            sage: shapes = vector(ComplexIntervalField(53), [ 0.5+0.8j, 0.5+0.8j, 0.5+0.8j])
            sage: for i in range(15):
            ...       penultimateShapes = shapes
            ...       centers = [ shape.center() for shape in shapes ]
            ...       C = KrawczykShapesEngine(M, centers, bits_prec = 53)
            ...       shapes = C.krawczyk_interval(shapes)
            sage: shapes # doctest: +NUMERIC12
            (0.78055252785073? + 0.91447366296773?*I, 0.780552527850725? + 0.91447366296773?*I, 0.460021175573718? + 0.632624193605256?*I)

        """
    derivative = self.log_gluing_LHS_derivatives_sparse(shape_intervals)
    p = KrawczykShapesEngine.matrix_times_sparse(self.approx_inverse, derivative)
    diff = self.identity - p
    return self.first_term + diff * (shape_intervals - self.initial_shapes)