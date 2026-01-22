from __future__ import annotations
import os
import re
from ..io import (
from ..util import (
from .common import (
from ..data import (
from ..target import (
def extract_powershell_module_utils_imports(path: str, module_utils: set[str]) -> set[str]:
    """Return a set of module_utils imports found in the specified source file."""
    imports = set()
    code = read_text_file(path)
    if data_context().content.is_ansible and '# POWERSHELL_COMMON' in code:
        imports.add('Ansible.ModuleUtils.Legacy')
    lines = code.splitlines()
    line_number = 0
    for line in lines:
        line_number += 1
        match = re.search('(?i)^#\\s*(?:requires\\s+-modules?|ansiblerequires\\s+-powershell)\\s*((?:Ansible|ansible_collections|\\.)\\..+)', line)
        if not match:
            continue
        import_name = resolve_csharp_ps_util(match.group(1), path)
        if import_name in module_utils:
            imports.add(import_name)
        elif data_context().content.is_ansible or import_name.startswith('ansible_collections.%s' % data_context().content.prefix):
            display.warning('%s:%d Invalid module_utils import: %s' % (path, line_number, import_name))
    return imports