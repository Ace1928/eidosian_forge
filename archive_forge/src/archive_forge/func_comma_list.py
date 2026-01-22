from typing import Any, List
def comma_list(items: List[Any]) -> str:
    """Convert a list to a comma-separated string."""
    return ', '.join((str(item) for item in items))