import argparse
import functools
import itertools
import marshal
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import List
def write_main(self, install_root, oss, symbol_name):
    """Write the `main.c` file containing a table enumerating all the frozen modules."""
    with open(os.path.join(install_root, 'main.c'), 'w') as outfp:
        outfp.write(MAIN_INCLUDES)
        for m in self.frozen_modules:
            outfp.write(f'extern unsigned char {m.c_name}[];\n')
        outfp.write(MAIN_PREFIX_TEMPLATE.format(symbol_name))
        for m in self.frozen_modules:
            outfp.write(f'\t{{"{m.module_name}", {m.c_name}, {m.size}}},\n')
        outfp.write(MAIN_SUFFIX)
        if oss:
            outfp.write(FAKE_PREFIX)
            outfp.write(MAIN_SUFFIX)