from sympy.matrices.dense import diag
from sympy.diffgeom import (twoform_to_matrix,
import sympy.diffgeom.rn
from sympy.tensor.array import ImmutableDenseNDimArray

unit test describing the hyperbolic half-plane with the Poincare metric. This
is a basic model of hyperbolic geometry on the (positive) half-space

{(x,y) \in R^2 | y > 0}

with the Riemannian metric

ds^2 = (dx^2 + dy^2)/y^2

It has constant negative scalar curvature = -2

https://en.wikipedia.org/wiki/Poincare_half-plane_model
