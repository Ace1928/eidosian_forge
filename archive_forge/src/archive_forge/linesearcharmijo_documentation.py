import logging
import math
import numpy as np
from ase.utils import longsum
Verify passed parameters and set appropriate attributes accordingly.

        A suitable value for the initial step-length guess will be either
        verified or calculated, stored in the attribute self.a_start, and
        returned.

        Args:
            The args should be identical to those of self.run().

        Returns:
            The suitable initial step-length guess a_start

        Raises:
            ValueError for problems with arguments

        