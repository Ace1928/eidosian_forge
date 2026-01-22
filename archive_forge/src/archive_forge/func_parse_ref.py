from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_ref(container: Union['Repo', 'RefsContainer'], refspec: Union[str, bytes]) -> 'Ref':
    """Parse a string referring to a reference.

    Args:
      container: A RefsContainer object
      refspec: A string referring to a ref
    Returns: A ref
    Raises:
      KeyError: If the ref can not be found
    """
    refspec = to_bytes(refspec)
    possible_refs = [refspec, b'refs/' + refspec, b'refs/tags/' + refspec, b'refs/heads/' + refspec, b'refs/remotes/' + refspec, b'refs/remotes/' + refspec + b'/HEAD']
    for ref in possible_refs:
        if ref in container:
            return ref
    raise KeyError(refspec)