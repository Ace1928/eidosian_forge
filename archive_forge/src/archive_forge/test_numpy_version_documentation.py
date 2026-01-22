import re
import numpy as np
from numpy.testing import assert_

Check the numpy version is valid.

Note that a development version is marked by the presence of 'dev0' or '+'
in the version string, all else is treated as a release. The version string
itself is set from the output of ``git describe`` which relies on tags.

Examples
--------

Valid Development: 1.22.0.dev0 1.22.0.dev0+5-g7999db4df2 1.22.0+5-g7999db4df2
Valid Release: 1.21.0.rc1, 1.21.0.b1, 1.21.0
Invalid: 1.22.0.dev, 1.22.0.dev0-5-g7999db4dfB, 1.21.0.d1, 1.21.a

Note that a release is determined by the version string, which in turn
is controlled by the result of the ``git describe`` command.
