from __future__ import absolute_import
import cython
from collections import defaultdict
import json
import operator
import os
import re
import sys
from .PyrexTypes import CPtrType
from . import Future
from . import Annotate
from . import Code
from . import Naming
from . import Nodes
from . import Options
from . import TypeSlots
from . import PyrexTypes
from . import Pythran
from .Errors import error, warning, CompileError
from .PyrexTypes import py_object_type
from ..Utils import open_new_file, replace_suffix, decode_filename, build_hex_version, is_cython_generated_file
from .Code import UtilityCode, IncludeCode, TempitaUtilityCode
from .StringEncoding import EncodedString, encoded_string_or_bytes_literal
from .Pythran import has_np_pythran
def generate_main_method(self, env, code):
    module_is_main = self.is_main_module_flag_cname()
    if Options.embed == 'main':
        wmain = 'wmain'
    else:
        wmain = Options.embed
    main_method = UtilityCode.load_cached('MainFunction', 'Embed.c')
    code.globalstate.use_utility_code(main_method.specialize(module_name=env.module_name, module_is_main=module_is_main, main_method=Options.embed, wmain_method=wmain))