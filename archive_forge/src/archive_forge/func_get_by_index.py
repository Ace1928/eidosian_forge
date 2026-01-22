from collections import deque
import logging
from .exceptions import InvalidTableIndex
def get_by_index(self, index):
    """
        Returns the entry specified by index

        Note that the table is 1-based ie an index of 0 is
        invalid.  This is due to the fact that a zero value
        index signals that a completely unindexed header
        follows.

        The entry will either be from the static table or
        the dynamic table depending on the value of index.
        """
    original_index = index
    index -= 1
    if 0 <= index:
        if index < HeaderTable.STATIC_TABLE_LENGTH:
            return HeaderTable.STATIC_TABLE[index]
        index -= HeaderTable.STATIC_TABLE_LENGTH
        if index < len(self.dynamic_entries):
            return self.dynamic_entries[index]
    raise InvalidTableIndex('Invalid table index %d' % original_index)