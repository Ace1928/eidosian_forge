import contextlib
import functools
import io
import os
import shutil
import subprocess
import sys
import sysconfig
import setuptools
@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv('TRITON_LIBCUDA_PATH')
    if env_libcuda_path:
        return [env_libcuda_path]
    libs = subprocess.check_output(['/sbin/ldconfig', '-p']).decode()
    locs = [line.split()[-1] for line in libs.splitlines() if 'libcuda.so' in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv('LD_LIBRARY_PATH')
    if env_ld_library_path and (not dirs):
        dirs = [dir for dir in env_ld_library_path.split(':') if os.path.exists(os.path.join(dir, 'libcuda.so'))]
    msg = 'libcuda.so cannot found!\n'
    if locs:
        msg += 'Possible files are located at %s.' % str(locs)
        msg += 'Please create a symlink of libcuda.so to any of the file.'
    else:
        msg += 'Please make sure GPU is setup and then run "/sbin/ldconfig"'
        msg += ' (requires sudo) to refresh the linker cache.'
    assert any((os.path.exists(os.path.join(path, 'libcuda.so')) for path in dirs)), msg
    return dirs