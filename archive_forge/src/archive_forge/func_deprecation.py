import enum
import typing as T
@property
def deprecation(self) -> T.Optional[DocstringDeprecated]:
    """Return a single information on function deprecation notes."""
    for item in self.meta:
        if isinstance(item, DocstringDeprecated):
            return item
    return None