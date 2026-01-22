from collections import namedtuple
def get_primary_key_columns(self, cursor, table_name):
    """Return a list of primary key columns for the given table."""
    for constraint in self.get_constraints(cursor, table_name).values():
        if constraint['primary_key']:
            return constraint['columns']
    return None