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
def check_yaml(self, conf, path, contents):
    """Check the given YAML."""
    self.check_parsable(path, contents)
    self.messages += [self.result_to_message(r, path) for r in linter.run(contents, conf, path)]