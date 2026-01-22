from snappy.sage_helper import _within_sage


Giac [1] is included in Sage 6.8 (July 2015) and newer with Sage
providing a pexpect interpreter interface.  Additionally, there is a
Cython-based wrapper for Giac [2] with a Sage-specific incarnation [3]
which is installable as an optional Sage spkg::

  sage -i giacpy_sage

This module is to make it easy to access a version of Giac from within
Sage, preferring the Cython-based wrapper if available.

[1] https://www-fourier.ujf-grenoble.fr/~parisse/giac.html
[2] https://pypi.org/project/giacpy
[3] https://gitlab.math.univ-paris-diderot.fr/han/giacpy-sage

