import os
import sys
import shlex
import importlib
import subprocess
import pkg_resources
import pathlib
from typing import List, Type, Any, Union, Dict
from types import ModuleType

        Resolves the import based on params. 
        Will automatically assume as resolve_missing = True, require = True
        
        The naming scheme is resolved as
        
        - module.submodule:attribute
        - pip_name|module.submodule:attribute # if pip_name is not the same

        Importing a module:
        LazyLib['tensorflow'] -> tensorflow Module
        LazyLib['fusepy|fuse'] -> fuse Module
        
        Importing a submodule:
        LazyLib['tensorflow.io'] -> io submodule

        Importing something from within a submodule:
        LazyLib['tensorflow.io.gfile:GFile'] -> GFile class
        