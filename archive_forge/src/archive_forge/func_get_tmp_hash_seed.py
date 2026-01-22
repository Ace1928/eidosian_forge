import os
import sys
import tempfile
from IPython.core.compilerop import CachingCompiler
def get_tmp_hash_seed():
    """Get a temp hash seed."""
    return 3339675911