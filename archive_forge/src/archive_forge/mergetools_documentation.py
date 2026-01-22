import os
import shutil
import subprocess
import sys
import tempfile
from .lazy_import import lazy_import
from breezy import (
Invokes the given merge tool command line, substituting the given
    filename according to the embedded substitution markers. Optionally, it
    will use the given invoker function instead of the default
    subprocess_invoker.
    