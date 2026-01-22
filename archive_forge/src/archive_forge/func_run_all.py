from __future__ import annotations
from lazyops.libs.dbinit.base import Engine
from lazyops.libs.dbinit.entities.entity import Entity
from lazyops.libs.dbinit.entities.role import Role
@classmethod
def run_all(cls, engine: Engine | None=None) -> None:
    """
        Attempts to create all declared entities then grant all declared privileges. Main way to do so.
        :param engine: A :class:`sqlalchemy.Engine` that defines the connection to a Postgres instance/cluster.
        """
    cls._handle_engine(engine)
    cls.create_all()
    cls.grant_all()