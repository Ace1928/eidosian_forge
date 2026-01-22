import os
from editorconfig import VERSION
from editorconfig.exceptions import PathError, VersionError
from editorconfig.ini import EditorConfigParser
def check_assertions(self):
    """Raise error if filepath or version have invalid values"""
    if not os.path.isabs(self.filepath):
        raise PathError('Input file must be a full path name.')
    if self.version is not None and self.version[:3] > VERSION[:3]:
        raise VersionError('Required version is greater than the current version.')