from typing import List, Optional, Union
def return_fields(self, *fields) -> 'Query':
    """Add fields to return fields."""
    self._return_fields += fields
    return self