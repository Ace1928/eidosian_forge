import ast
import json
import sys
import urllib
from wandb_gql import gql
import wandb
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.apis.paginator import Paginator
from wandb.sdk.lib import ipython
def python_to_mongo(self, filterstr):
    try:
        tree = ast.parse(self._convert(filterstr), mode='eval')
    except SyntaxError as e:
        raise ValueError('Invalid python comparison expression; form something like `my_col == 123`') from e
    multiple_filters = hasattr(tree.body, 'op')
    if multiple_filters:
        op = self.AST_OPERATORS.get(type(tree.body.op))
        values = [self._handle_compare(v) for v in tree.body.values]
    else:
        op = '$and'
        values = [self._handle_compare(tree.body)]
    return {'$or': [{op: values}]}