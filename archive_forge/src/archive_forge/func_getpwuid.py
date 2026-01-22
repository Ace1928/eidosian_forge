from typing import List, Optional
def getpwuid(self, uid: int) -> _UserRecord:
    """
        Return the user record corresponding to the given uid.
        """
    for entry in self._users:
        if entry.pw_uid == uid:
            return entry
    raise KeyError()