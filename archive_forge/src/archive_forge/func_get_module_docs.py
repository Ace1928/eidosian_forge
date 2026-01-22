from __future__ import annotations
import ast
import json
import os
import re
import sys
import typing as t
import yaml
from yaml.resolver import Resolver
from yaml.constructor import SafeConstructor
from yaml.error import MarkedYAMLError
from yaml.cyaml import CParser
from yamllint import linter
from yamllint.config import YamlLintConfig
def get_module_docs(self, path, contents):
    """Return the module documentation for the given module contents."""
    module_doc_types = ['DOCUMENTATION', 'EXAMPLES', 'RETURN']
    docs = {}
    fmt_re = re.compile('^# fmt:\\s+(\\S+)')

    def check_assignment(statement, doc_types=None):
        """Check the given statement for a documentation assignment."""
        for target in statement.targets:
            if not isinstance(target, ast.Name):
                continue
            if doc_types and target.id not in doc_types:
                continue
            fmt_match = fmt_re.match(statement.value.value.lstrip())
            fmt = 'yaml'
            if fmt_match:
                fmt = fmt_match.group(1)
            docs[target.id] = dict(yaml=statement.value.value, lineno=statement.lineno, end_lineno=statement.lineno + len(statement.value.value.splitlines()), fmt=fmt.lower())
    module_ast = self.parse_module(path, contents)
    if not module_ast:
        return {}
    is_plugin = path.startswith('lib/ansible/modules/') or path.startswith('lib/ansible/plugins/') or path.startswith('plugins/')
    is_doc_fragment = path.startswith('lib/ansible/plugins/doc_fragments/') or path.startswith('plugins/doc_fragments/')
    if is_plugin and (not is_doc_fragment):
        for body_statement in module_ast.body:
            if isinstance(body_statement, ast.Assign):
                check_assignment(body_statement, module_doc_types)
    elif is_doc_fragment:
        for body_statement in module_ast.body:
            if isinstance(body_statement, ast.ClassDef):
                for class_statement in body_statement.body:
                    if isinstance(class_statement, ast.Assign):
                        check_assignment(class_statement)
    else:
        raise Exception('unsupported path: %s' % path)
    return docs