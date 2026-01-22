import os
from warnings import warn
import sys
from distutils.errors import DistutilsExecError
from distutils.spawn import spawn
from distutils.dir_util import mkpath
from distutils import log
def make_zipfile(base_name, base_dir, verbose=0, dry_run=0):
    """Create a zip file from all the files under 'base_dir'.

    The output zip file will be named 'base_name' + ".zip".  Uses either the
    "zipfile" Python module (if available) or the InfoZIP "zip" utility
    (if installed and found on the default search path).  If neither tool is
    available, raises DistutilsExecError.  Returns the name of the output zip
    file.
    """
    zip_filename = base_name + '.zip'
    mkpath(os.path.dirname(zip_filename), dry_run=dry_run)
    if zipfile is None:
        if verbose:
            zipoptions = '-r'
        else:
            zipoptions = '-rq'
        try:
            spawn(['zip', zipoptions, zip_filename, base_dir], dry_run=dry_run)
        except DistutilsExecError:
            raise DistutilsExecError("unable to create zip file '%s': could neither import the 'zipfile' module nor find a standalone zip utility" % zip_filename)
    else:
        log.info("creating '%s' and adding '%s' to it", zip_filename, base_dir)
        if not dry_run:
            try:
                zip = zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED)
            except RuntimeError:
                zip = zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_STORED)
            with zip:
                if base_dir != os.curdir:
                    path = os.path.normpath(os.path.join(base_dir, ''))
                    zip.write(path, path)
                    log.info("adding '%s'", path)
                for dirpath, dirnames, filenames in os.walk(base_dir):
                    for name in dirnames:
                        path = os.path.normpath(os.path.join(dirpath, name, ''))
                        zip.write(path, path)
                        log.info("adding '%s'", path)
                    for name in filenames:
                        path = os.path.normpath(os.path.join(dirpath, name))
                        if os.path.isfile(path):
                            zip.write(path, path)
                            log.info("adding '%s'", path)
    return zip_filename