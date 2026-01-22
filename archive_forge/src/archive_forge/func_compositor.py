import sys
import time
from IPython.core.magic import Magics, line_cell_magic, line_magic, magics_class
from IPython.display import HTML, display
from ..core.options import Options, Store, StoreOptions, options_policy
from ..core.pprint import InfoPrinter
from ..operation import Compositor
from IPython.core import page
@line_magic
def compositor(self, line):
    if line.strip():
        for definition in CompositorSpec.parse(line.strip(), ns=self.shell.user_ns):
            group = {group: Options() for group in Options._option_groups}
            type_name = definition.output_type.__name__
            Store.options()[type_name + '.' + definition.group] = group
            Compositor.register(definition)
    else:
        print('For help with the %compositor magic, call %compositor?\n')