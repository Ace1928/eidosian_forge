from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import importlib
import os
import sys
import fire
def import_from_file_path(path):
    """Performs a module import given the filename.

  Args:
    path (str): the path to the file to be imported.

  Raises:
    IOError: if the given file does not exist or importlib fails to load it.

  Returns:
    Tuple[ModuleType, str]: returns the imported module and the module name,
      usually extracted from the path itself.
  """
    if not os.path.exists(path):
        raise IOError('Given file path does not exist.')
    module_name = os.path.basename(path)
    if sys.version_info.major == 3 and sys.version_info.minor < 5:
        loader = importlib.machinery.SourceFileLoader(fullname=module_name, path=path)
        module = loader.load_module(module_name)
    elif sys.version_info.major == 3:
        from importlib import util
        spec = util.spec_from_file_location(module_name, path)
        if spec is None:
            raise IOError('Unable to load module from specified path.')
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        import imp
        module = imp.load_source(module_name, path)
    return (module, module_name)