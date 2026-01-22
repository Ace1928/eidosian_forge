import sqlite3
from typing import (
@row_factory.setter
def row_factory(self, factory: Optional[Type]) -> None:
    self._cursor.row_factory = factory