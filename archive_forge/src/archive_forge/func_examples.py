import enum
import typing as T
@property
def examples(self) -> T.List[DocstringExample]:
    """Return a list of information on function examples."""
    return [item for item in self.meta if isinstance(item, DocstringExample)]