import io
import os
import re
import sys
import time
import socket
import base64
import tempfile
import logging
from pyomo.common.dependencies import attempt_import

        Read in the kestrel_options to pick out the solver name.
        The tricky parts:
          we don't want to be case sensitive, but NEOS is.
          we need to read in options variable
        