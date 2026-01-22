from __future__ import absolute_import
import math, sys
def verify_slice(s):
    if s.start or s.stop or s.step not in (None, 1):
        raise InvalidTypeSpecification('Only a step of 1 may be provided to indicate C or Fortran contiguity')