from typing import Any, Dict, Optional, Sequence
def row_count_query(self, query: str) -> str:
    """
        Get a query that gives the names of rows that `query` would produce.

        Parameters
        ----------
        query : str
            The SQL query to check.

        Returns
        -------
        str
        """
    return f'SELECT COUNT(*) FROM ({query}) AS _MODIN_COUNT_QUERY'