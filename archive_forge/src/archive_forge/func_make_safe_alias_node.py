from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import copy
import logging
from pasta.augment import errors
from pasta.base import ast_utils
from pasta.base import scope
def make_safe_alias_node(alias_name, asname):
    new_alias = ast.alias(name=alias_name, asname=asname)
    imported_name = asname or alias_name
    counter = 0
    while imported_name in sc.names:
        counter += 1
        imported_name = new_alias.asname = '%s_%d' % (asname or alias_name, counter)
    return new_alias