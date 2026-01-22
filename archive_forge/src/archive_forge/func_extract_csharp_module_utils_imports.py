from __future__ import annotations
import os
import re
from ..io import (
from ..util import (
from .common import (
from ..data import (
from ..target import (
def extract_csharp_module_utils_imports(path: str, module_utils: set[str], is_pure_csharp: bool) -> set[str]:
    """Return a set of module_utils imports found in the specified source file."""
    imports = set()
    if is_pure_csharp:
        pattern = re.compile('(?i)^using\\s((?:Ansible|AnsibleCollections)\\..+);$')
    else:
        pattern = re.compile('(?i)^#\\s*ansiblerequires\\s+-csharputil\\s+((?:Ansible|ansible.collections|\\.)\\..+)')
    with open_text_file(path) as module_file:
        for line_number, line in enumerate(module_file, 1):
            match = re.search(pattern, line)
            if not match:
                continue
            import_name = resolve_csharp_ps_util(match.group(1), path)
            if import_name in module_utils:
                imports.add(import_name)
            elif data_context().content.is_ansible or import_name.startswith('ansible_collections.%s' % data_context().content.prefix):
                display.warning('%s:%d Invalid module_utils import: %s' % (path, line_number, import_name))
    return imports