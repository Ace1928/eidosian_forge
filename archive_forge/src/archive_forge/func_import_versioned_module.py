import sys
import traceback
def import_versioned_module(module, version, submodule=None):
    """Import a versioned module in format {module}.v{version][.{submodule}].

    :param module: the module name.
    :param version: the version number.
    :param submodule: the submodule name.
    :raises ValueError: For any invalid input.

    .. versionadded:: 0.3

    .. versionchanged:: 3.17
       Added *module* parameter.
    """
    if '.' in '%s' % version:
        raise ValueError("Parameter version shouldn't include character '.'.")
    module_str = '%s.v%s' % (module, version)
    if submodule:
        module_str = '.'.join((module_str, submodule))
    return import_module(module_str)