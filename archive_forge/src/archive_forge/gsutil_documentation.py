from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import os
import sys
import warnings
Reloads the google module to prefer our third_party copy.

  When Python is not invoked with the -S option, it may preload the google module via .pth file.
  This leads to the "site_packages" version being preferred over gsutil "third_party" version.
  To force the "third_party" version, insert the path at the start of sys.path and reload the google module.

  This is a hacky. Reloading is required for the rare case that users have
  google-auth already installed in their Python environment.
  Note that this reload may cause an issue for Python 3.5.3 and lower
  because of the weakref issue, fixed in Python 3.5.4:
  https://github.com/python/cpython/commit/9cd7e17640a49635d1c1f8c2989578a8fc2c1de6.
  