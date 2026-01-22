import copy
import os
import shutil
import warnings
import sys
import importlib
import uuid
import hashlib
from ._all_keywords import julia_keywords
from ._py_components_generation import reorder_props
def base_package_uid(project_shortname):
    return jl_dash_base_uuid if is_core_package(project_shortname) else jl_dash_uuid