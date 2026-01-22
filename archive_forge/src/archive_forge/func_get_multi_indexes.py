from the proposed insertion.   These values are specified using the
from __future__ import annotations
from collections import defaultdict
from functools import lru_cache
import re
from typing import Any
from typing import cast
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
from . import arraylib as _array
from . import json as _json
from . import pg_catalog
from . import ranges as _ranges
from .ext import _regconfig_fn
from .ext import aggregate_order_by
from .hstore import HSTORE
from .named_types import CreateDomainType as CreateDomainType  # noqa: F401
from .named_types import CreateEnumType as CreateEnumType  # noqa: F401
from .named_types import DOMAIN as DOMAIN  # noqa: F401
from .named_types import DropDomainType as DropDomainType  # noqa: F401
from .named_types import DropEnumType as DropEnumType  # noqa: F401
from .named_types import ENUM as ENUM  # noqa: F401
from .named_types import NamedType as NamedType  # noqa: F401
from .types import _DECIMAL_TYPES  # noqa: F401
from .types import _FLOAT_TYPES  # noqa: F401
from .types import _INT_TYPES  # noqa: F401
from .types import BIT as BIT
from .types import BYTEA as BYTEA
from .types import CIDR as CIDR
from .types import CITEXT as CITEXT
from .types import INET as INET
from .types import INTERVAL as INTERVAL
from .types import MACADDR as MACADDR
from .types import MACADDR8 as MACADDR8
from .types import MONEY as MONEY
from .types import OID as OID
from .types import PGBit as PGBit  # noqa: F401
from .types import PGCidr as PGCidr  # noqa: F401
from .types import PGInet as PGInet  # noqa: F401
from .types import PGInterval as PGInterval  # noqa: F401
from .types import PGMacAddr as PGMacAddr  # noqa: F401
from .types import PGMacAddr8 as PGMacAddr8  # noqa: F401
from .types import PGUuid as PGUuid
from .types import REGCLASS as REGCLASS
from .types import REGCONFIG as REGCONFIG  # noqa: F401
from .types import TIME as TIME
from .types import TIMESTAMP as TIMESTAMP
from .types import TSVECTOR as TSVECTOR
from ... import exc
from ... import schema
from ... import select
from ... import sql
from ... import util
from ...engine import characteristics
from ...engine import default
from ...engine import interfaces
from ...engine import ObjectKind
from ...engine import ObjectScope
from ...engine import reflection
from ...engine import URL
from ...engine.reflection import ReflectionDefaults
from ...sql import bindparam
from ...sql import coercions
from ...sql import compiler
from ...sql import elements
from ...sql import expression
from ...sql import roles
from ...sql import sqltypes
from ...sql import util as sql_util
from ...sql.compiler import InsertmanyvaluesSentinelOpts
from ...sql.visitors import InternalTraversal
from ...types import BIGINT
from ...types import BOOLEAN
from ...types import CHAR
from ...types import DATE
from ...types import DOUBLE_PRECISION
from ...types import FLOAT
from ...types import INTEGER
from ...types import NUMERIC
from ...types import REAL
from ...types import SMALLINT
from ...types import TEXT
from ...types import UUID as UUID
from ...types import VARCHAR
from ...util.typing import TypedDict
def get_multi_indexes(self, connection, schema, filter_names, scope, kind, **kw):
    table_oids = self._get_table_oids(connection, schema, filter_names, scope, kind, **kw)
    indexes = defaultdict(list)
    default = ReflectionDefaults.indexes
    batches = list(table_oids)
    while batches:
        batch = batches[0:3000]
        batches[0:3000] = []
        result = connection.execute(self._index_query, {'oids': [r[0] for r in batch]}).mappings()
        result_by_oid = defaultdict(list)
        for row_dict in result:
            result_by_oid[row_dict['indrelid']].append(row_dict)
        for oid, table_name in batch:
            if oid not in result_by_oid:
                indexes[schema, table_name] = default()
                continue
            for row in result_by_oid[oid]:
                index_name = row['relname_index']
                table_indexes = indexes[schema, table_name]
                all_elements = row['elements']
                all_elements_is_expr = row['elements_is_expr']
                indnkeyatts = row['indnkeyatts']
                if indnkeyatts and len(all_elements) > indnkeyatts:
                    inc_cols = all_elements[indnkeyatts:]
                    idx_elements = all_elements[:indnkeyatts]
                    idx_elements_is_expr = all_elements_is_expr[:indnkeyatts]
                    assert all((not is_expr for is_expr in all_elements_is_expr[indnkeyatts:]))
                else:
                    idx_elements = all_elements
                    idx_elements_is_expr = all_elements_is_expr
                    inc_cols = []
                index = {'name': index_name, 'unique': row['indisunique']}
                if any(idx_elements_is_expr):
                    index['column_names'] = [None if is_expr else expr for expr, is_expr in zip(idx_elements, idx_elements_is_expr)]
                    index['expressions'] = idx_elements
                else:
                    index['column_names'] = idx_elements
                sorting = {}
                for col_index, col_flags in enumerate(row['indoption']):
                    col_sorting = ()
                    if col_flags & 1:
                        col_sorting += ('desc',)
                        if not col_flags & 2:
                            col_sorting += ('nulls_last',)
                    elif col_flags & 2:
                        col_sorting += ('nulls_first',)
                    if col_sorting:
                        sorting[idx_elements[col_index]] = col_sorting
                if sorting:
                    index['column_sorting'] = sorting
                if row['has_constraint']:
                    index['duplicates_constraint'] = index_name
                dialect_options = {}
                if row['reloptions']:
                    dialect_options['postgresql_with'] = dict([option.split('=') for option in row['reloptions']])
                amname = row['amname']
                if amname != 'btree':
                    dialect_options['postgresql_using'] = row['amname']
                if row['filter_definition']:
                    dialect_options['postgresql_where'] = row['filter_definition']
                if self.server_version_info >= (11,):
                    index['include_columns'] = inc_cols
                    dialect_options['postgresql_include'] = inc_cols
                if row['indnullsnotdistinct']:
                    dialect_options['postgresql_nulls_not_distinct'] = row['indnullsnotdistinct']
                if dialect_options:
                    index['dialect_options'] = dialect_options
                table_indexes.append(index)
    return indexes.items()