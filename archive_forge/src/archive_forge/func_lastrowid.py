import sqlite3
from typing import (
@property
def lastrowid(self) -> Optional[int]:
    return self._cursor.lastrowid