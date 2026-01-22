from typing import Iterator, Set, Type
def get_all_subclasses_iterator(cls: Type) -> Iterator[Type]:
    """Iterate over all subclasses."""

    def recurse(cl: Type) -> Iterator[Type]:
        for subclass in cl.__subclasses__():
            yield subclass
            yield from recurse(subclass)
    yield from recurse(cls)