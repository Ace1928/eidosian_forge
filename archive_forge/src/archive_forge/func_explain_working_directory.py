from __future__ import annotations
import collections.abc as c
import dataclasses
import os
import typing as t
from .util import (
from .provider import (
from .provider.source import (
from .provider.source.unversioned import (
from .provider.source.installed import (
from .provider.source.unsupported import (
from .provider.layout import (
from .provider.layout.unsupported import (
def explain_working_directory(self) -> str:
    """Return a message explaining the working directory requirements."""
    blocks = ['The current working directory must be within the source tree being tested.', '']
    if ANSIBLE_SOURCE_ROOT:
        blocks.append(f'Testing Ansible: {ANSIBLE_SOURCE_ROOT}/')
        blocks.append('')
    cwd = os.getcwd()
    blocks.append('Testing an Ansible collection: {...}/ansible_collections/{namespace}/{collection}/')
    blocks.append('Example #1: community.general -> ~/code/ansible_collections/community/general/')
    blocks.append('Example #2: ansible.util -> ~/.ansible/collections/ansible_collections/ansible/util/')
    blocks.append('')
    blocks.append(f'Current working directory: {cwd}/')
    if os.path.basename(os.path.dirname(cwd)) == 'ansible_collections':
        blocks.append(f'Expected parent directory: {os.path.dirname(cwd)}/{{namespace}}/{{collection}}/')
    elif os.path.basename(cwd) == 'ansible_collections':
        blocks.append(f'Expected parent directory: {cwd}/{{namespace}}/{{collection}}/')
    elif 'ansible_collections' not in cwd.split(os.path.sep):
        blocks.append('No "ansible_collections" parent directory was found.')
    if isinstance(self.content.unsupported, list):
        blocks.extend(self.content.unsupported)
    message = '\n'.join(blocks)
    return message