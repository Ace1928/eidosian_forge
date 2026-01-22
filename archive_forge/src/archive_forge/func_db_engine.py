from __future__ import annotations
from typing import Sequence
from lazyops.libs.dbinit.base import Engine, TextClause, create_engine, text
from lazyops.libs.dbinit.data_structures.grant_to import GrantTo
from lazyops.libs.dbinit.data_structures.privileges import Privilege
from lazyops.libs.dbinit.entities.cluster_entity import ClusterEntity
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
from lazyops.libs.dbinit.mixins.grantable import Grantable
def db_engine(self) -> Engine:
    """
        Getter for this database's engine. Will create it if it is not yet set.
        :return: A `sqlalchemy.Engine` to connect to this database.
        """
    if not self._db_engine:
        host = self.__class__.engine().url.host
        port = self.__class__.engine().url.port
        user = self.__class__.engine().url.username
        pw = self.__class__.engine().url.password
        self._db_engine = create_engine(f'postgresql+psycopg://{user}:{pw}@{host}:{port}/{self.name}')
    return self._db_engine