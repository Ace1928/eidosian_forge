from __future__ import (absolute_import, division, print_function)
import os
import re
import shutil
import tempfile
import types
from ansible.module_utils.six.moves import configparser

            Return a list of RhsmPools whose name matches the provided regular expression
        