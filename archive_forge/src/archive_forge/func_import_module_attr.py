import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType
@classmethod
def import_module_attr(cls, name: str, module_name: str, library: str=None, pip_name: str=None, resolve_missing: bool=True, require: bool=False, upgrade: bool=False) -> Any:
    """ Lazily resolves libs and imports the name, aliasing
            immportlib.import_module
            Returns an attribute from the module

            ie ->   LazyLib.import_module_attr('GFile', 'tensorflow.io.gfile', 'tensorflow') # if tensorflow is not expected to be available
                    LazyLib.import_module_attr('GFile', 'tensorflow.io.gfile') # if tensorflow is expected to be available
            returns GFile
        """
    mod = cls.import_module(name=module_name, library=library, pip_name=pip_name, resolve_missing=resolve_missing, require=require, upgrade=upgrade)
    return getattr(mod, name)