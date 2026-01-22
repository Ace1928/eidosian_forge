from typing import Any, Dict, Optional, TypeVar
from ...public import Api as PublicApi
from ...public import PythonMongoishQueryGenerator, QueryGenerator, Runs
from .util import Attr, Base, coalesce, generate_name, nested_get, nested_set
def set_filters_with_python_expr(self, expr):
    self.filters = self.pm_query_generator.python_to_mongo(expr)
    return self