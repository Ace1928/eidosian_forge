import os
from math import ceil
from typing import Any, Dict, List, Optional
def set_db(self, db: Any) -> None:
    from arango.database import Database
    if not isinstance(db, Database):
        msg = '**db** parameter must inherit from arango.database.Database'
        raise TypeError(msg)
    self.__db: Database = db
    self.set_schema()