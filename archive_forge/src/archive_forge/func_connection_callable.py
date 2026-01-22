from __future__ import annotations
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
from .. import event
from .. import exc
from .. import inspect
from .. import util
from ..orm import PassiveFlag
from ..orm._typing import OrmExecuteOptionsParameter
from ..orm.interfaces import ORMOption
from ..orm.mapper import Mapper
from ..orm.query import Query
from ..orm.session import _BindArguments
from ..orm.session import _PKIdentityArgument
from ..orm.session import Session
from ..util.typing import Protocol
from ..util.typing import Self
def connection_callable(self, mapper: Optional[Mapper[_T]]=None, instance: Optional[Any]=None, shard_id: Optional[ShardIdentifier]=None, **kw: Any) -> Connection:
    """Provide a :class:`_engine.Connection` to use in the unit of work
        flush process.

        """
    if shard_id is None:
        shard_id = self._choose_shard_and_assign(mapper, instance)
    if self.in_transaction():
        trans = self.get_transaction()
        assert trans is not None
        return trans.connection(mapper, shard_id=shard_id)
    else:
        bind = self.get_bind(mapper=mapper, shard_id=shard_id, instance=instance)
        if isinstance(bind, Engine):
            return bind.connect(**kw)
        else:
            assert isinstance(bind, Connection)
            return bind