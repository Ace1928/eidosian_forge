from typing import List, Optional, Union
def limit_fields(self, *fields: List[str]) -> 'Query':
    """
        Limit the search to specific TEXT fields only.

        - **fields**: A list of strings, case sensitive field names
        from the defined schema.
        """
    self._fields = fields
    return self