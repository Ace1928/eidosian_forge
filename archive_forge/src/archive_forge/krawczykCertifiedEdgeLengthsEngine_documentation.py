from .hyperbolicStructure import *
from .verificationError import *
from sage.all import RealDoubleField, RealIntervalField, vector, matrix, pi

        Given two vectors of intervals, return whether the first one
        is contained in the second one.  Examples::

            sage: RIF = RealIntervalField(80)
            sage: CIF = ComplexIntervalField(80)
            sage: box = CIF(RIF(-1,1),RIF(-1,1))
            sage: a = [ CIF(0.1), CIF(1) + box ]
            sage: b = [ CIF(0) + box, CIF(1) + 2 * box ]
            sage: c = [ CIF(0), CIF(1) + 3 * box ]

            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(a, b)
            True
            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(a, c)
            False
            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(b, a)
            False
            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(b, c)
            False
            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(c, a)
            False
            sage: KrawczykCertifiedShapesEngine.interval_vector_is_contained_in(c, b)
            False
        