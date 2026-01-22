import itertools
from ..snap.t3mlite.simplex import *
from .rational_linear_algebra import QQ, Vector3, Vector4, Matrix
from .barycentric_geometry import (BarycentricPoint,
from .mcomplex_with_link import McomplexWithLink

    >>> K = example10()
    >>> components = embed_link_in_S3(K)
    >>> len(components), len(components[0])
    (1, 19)
    