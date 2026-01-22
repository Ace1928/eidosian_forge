import setuptools
from setuptools.command.build_ext import build_ext
from setuptools.dist import Distribution
import numpy as np
import functools
import os
import subprocess
import sys
from tempfile import mkdtemp
from contextlib import contextmanager
from pathlib import Path
@functools.lru_cache(maxsize=1)
def external_compiler_works():
    """
    Returns True if the "external compiler" bound in numpy.distutil is present
    and working, False otherwise.
    """
    compiler = new_compiler()
    customize_compiler(compiler)
    for suffix in ['.c', '.cxx']:
        try:
            with _gentmpfile(suffix) as ntf:
                simple_c = 'int main(void) { return 0; }'
                ntf.write(simple_c)
                ntf.flush()
                ntf.close()
                compiler.compile([ntf.name], output_dir=Path(ntf.name).anchor)
        except Exception:
            return False
    return True