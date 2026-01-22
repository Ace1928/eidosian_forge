from __future__ import annotations
import contextlib
from dataclasses import dataclass
from enum import auto
from enum import Flag
from enum import unique
from typing import Any
from typing import Callable
from typing import Collection
from typing import Dict
from typing import Generator
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .base import Connection
from .base import Engine
from .. import exc
from .. import inspection
from .. import sql
from .. import util
from ..sql import operators
from ..sql import schema as sa_schema
from ..sql.cache_key import _ad_hoc_cache_key_from_args
from ..sql.elements import TextClause
from ..sql.type_api import TypeEngine
from ..sql.visitors import InternalTraversal
from ..util import topological
from ..util.typing import final
def sort_tables_on_foreign_key_dependency(self, consider_schemas: Collection[Optional[str]]=(None,), **kw: Any) -> List[Tuple[Optional[Tuple[Optional[str], str]], List[Tuple[Tuple[Optional[str], str], Optional[str]]]]]:
    """Return dependency-sorted table and foreign key constraint names
        referred to within multiple schemas.

        This method may be compared to
        :meth:`.Inspector.get_sorted_table_and_fkc_names`, which
        works on one schema at a time; here, the method is a generalization
        that will consider multiple schemas at once including that it will
        resolve for cross-schema foreign keys.

        .. versionadded:: 2.0

        """
    SchemaTab = Tuple[Optional[str], str]
    tuples: Set[Tuple[SchemaTab, SchemaTab]] = set()
    remaining_fkcs: Set[Tuple[SchemaTab, Optional[str]]] = set()
    fknames_for_table: Dict[SchemaTab, Set[Optional[str]]] = {}
    tnames: List[SchemaTab] = []
    for schname in consider_schemas:
        schema_fkeys = self.get_multi_foreign_keys(schname, **kw)
        tnames.extend(schema_fkeys)
        for (_, tname), fkeys in schema_fkeys.items():
            fknames_for_table[schname, tname] = {fk['name'] for fk in fkeys}
            for fkey in fkeys:
                if tname != fkey['referred_table'] or schname != fkey['referred_schema']:
                    tuples.add(((fkey['referred_schema'], fkey['referred_table']), (schname, tname)))
    try:
        candidate_sort = list(topological.sort(tuples, tnames))
    except exc.CircularDependencyError as err:
        edge: Tuple[SchemaTab, SchemaTab]
        for edge in err.edges:
            tuples.remove(edge)
            remaining_fkcs.update(((edge[1], fkc) for fkc in fknames_for_table[edge[1]]))
        candidate_sort = list(topological.sort(tuples, tnames))
    ret: List[Tuple[Optional[SchemaTab], List[Tuple[SchemaTab, Optional[str]]]]]
    ret = [((schname, tname), [((schname, tname), fk) for fk in fknames_for_table[schname, tname].difference((name for _, name in remaining_fkcs))]) for schname, tname in candidate_sort]
    return ret + [(None, list(remaining_fkcs))]