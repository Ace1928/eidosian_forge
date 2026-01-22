import contextlib
import functools
import hashlib
import os
import re
import sys
import textwrap
from argparse import Namespace
from dataclasses import fields, is_dataclass
from enum import auto, Enum
from typing import (
from typing_extensions import Self
from torchgen.code_template import CodeTemplate
@staticmethod
def from_namespaced_entity(namespaced_entity: str, max_level: int=2) -> 'NamespaceHelper':
    """
        Generate helper from nested namespaces as long as class/function name. E.g.: "torch::lazy::add"
        """
    names = namespaced_entity.split('::')
    entity_name = names[-1]
    namespace_str = '::'.join(names[:-1])
    return NamespaceHelper(namespace_str=namespace_str, entity_name=entity_name, max_level=max_level)