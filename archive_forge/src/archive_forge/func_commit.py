from .reference import Reference
from typing import Any, Type, Union, TYPE_CHECKING
from git.types import Commit_ish, PathLike
@property
def commit(self) -> 'Commit':
    """:return: Commit object the tag ref points to

        :raise ValueError: If the tag points to a tree or blob
        """
    obj = self.object
    while obj.type != 'commit':
        if obj.type == 'tag':
            obj = obj.object
        else:
            raise ValueError(('Cannot resolve commit as tag %s points to a %s object - ' + 'use the `.object` property instead to access it') % (self, obj.type))
    return obj