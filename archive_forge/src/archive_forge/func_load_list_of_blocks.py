from __future__ import (absolute_import, division, print_function)
import os
from ansible import constants as C
from ansible.errors import AnsibleParserError, AnsibleUndefinedVariable, AnsibleAssertionError
from ansible.module_utils.common.text.converters import to_native
from ansible.parsing.mod_args import ModuleArgsParser
from ansible.utils.display import Display
def load_list_of_blocks(ds, play, parent_block=None, role=None, task_include=None, use_handlers=False, variable_manager=None, loader=None):
    """
    Given a list of mixed task/block data (parsed from YAML),
    return a list of Block() objects, where implicit blocks
    are created for each bare Task.
    """
    from ansible.playbook.block import Block
    if not isinstance(ds, (list, type(None))):
        raise AnsibleAssertionError('%s should be a list or None but is %s' % (ds, type(ds)))
    block_list = []
    if ds:
        count = iter(range(len(ds)))
        for i in count:
            block_ds = ds[i]
            implicit_blocks = []
            while block_ds is not None and (not Block.is_block(block_ds)):
                implicit_blocks.append(block_ds)
                i += 1
                next(count, None)
                try:
                    block_ds = ds[i]
                except IndexError:
                    block_ds = None
            for b in (implicit_blocks, block_ds):
                if b:
                    block_list.append(Block.load(b, play=play, parent_block=parent_block, role=role, task_include=task_include, use_handlers=use_handlers, variable_manager=variable_manager, loader=loader))
    return block_list