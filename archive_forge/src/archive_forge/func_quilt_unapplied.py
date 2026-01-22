import errno
import os
import signal
import subprocess
from ... import errors, osutils, trace
from ... import transport as _mod_transport
def quilt_unapplied(working_dir, patches_dir=None, series_file=None):
    """Find the list of unapplied quilt patches.

    :param working_dir: Directory to work in
    :param patches_dir: Optional patches directory
    :param series_file: Optional series file
    """
    working_dir = os.path.abspath(working_dir)
    if patches_dir is None:
        patches_dir = os.path.join(working_dir, DEFAULT_PATCHES_DIR)
    try:
        unapplied_patches = run_quilt(['unapplied'], working_dir=working_dir, patches_dir=patches_dir, series_file=series_file).splitlines()
        patch_names = []
        for patch in unapplied_patches:
            patch = os.fsdecode(patch)
            patch_names.append(os.path.relpath(patch, patches_dir))
        return patch_names
    except QuiltError as e:
        if e.retcode == 1:
            return []
        raise