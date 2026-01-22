import contextlib
import os
import pathlib
import shutil
import stat
import sys
import zipfile
import {module}
Run the zipapp command line interface.

    The ARGS parameter lets you specify the argument list directly.
    Omitting ARGS (or setting it to None) works as for argparse, using
    sys.argv[1:] as the argument list.
    