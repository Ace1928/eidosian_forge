import os
from typing import Dict, Type
def last_id(self, cursor, table):
    table = self.tname(table)
    sql = f"SELECT currval('{table}_pk_seq')"
    cursor.execute(sql)
    rv = cursor.fetchone()
    return rv[0]