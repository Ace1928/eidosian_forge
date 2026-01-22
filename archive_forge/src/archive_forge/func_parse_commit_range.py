from typing import TYPE_CHECKING, Iterator, List, Optional, Tuple, Union
def parse_commit_range(repo: 'Repo', committishs: Union[str, bytes]) -> Iterator['Commit']:
    """Parse a string referring to a range of commits.

    Args:
      repo: A `Repo` object
      committishs: A string referring to a range of commits.
    Returns: An iterator over `Commit` objects
    Raises:
      KeyError: When the reference commits can not be found
      ValueError: If the range can not be parsed
    """
    committishs = to_bytes(committishs)
    return iter([parse_commit(repo, committishs)])