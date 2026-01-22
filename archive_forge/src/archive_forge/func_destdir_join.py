from pathlib import PurePath
def destdir_join(d1: str, d2: str) -> str:
    if not d1:
        return d2
    return str(PurePath(d1, *PurePath(d2).parts[1:]))