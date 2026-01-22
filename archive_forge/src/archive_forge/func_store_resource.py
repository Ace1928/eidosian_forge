import inspect
import os
import sys
import sysconfig
import warnings
def store_resource(project, resource_path, filename):
    """ Store the content of a resource, given by the name of the project
        and the path (relative to the root of the project), into a newly
        created file.

        The first two arguments (project and resource_path) are the same
        as for the function find_resource in this module.  The third
        argument (filename) is the name of the file which will be created,
        or overwritten if it already exists.
        The return value in always None.

        .. deprecated:: 6.3.0
    """
    warnings.warn('store_resource is deprecated. Use importlib.resources instead.', DeprecationWarning, stacklevel=2)
    fi = find_resource(project, resource_path)
    if fi is None:
        raise RuntimeError('Resource not found for project "%s": %s' % (project, resource_path))
    with open(filename, 'wb') as fo:
        fo.write(fi.read())
    fi.close()