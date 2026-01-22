import os
from pathlib import Path
import sys
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import iniconfig
from .exceptions import UsageError
from _pytest.outcomes import fail
from _pytest.pathlib import absolutepath
from _pytest.pathlib import commonpath
from _pytest.pathlib import safe_exists
def determine_setup(*, inifile: Optional[str], args: Sequence[str], rootdir_cmd_arg: Optional[str], invocation_dir: Path) -> Tuple[Path, Optional[Path], Dict[str, Union[str, List[str]]]]:
    """Determine the rootdir, inifile and ini configuration values from the
    command line arguments.

    :param inifile:
        The `--inifile` command line argument, if given.
    :param args:
        The free command line arguments.
    :param rootdir_cmd_arg:
        The `--rootdir` command line argument, if given.
    :param invocation_dir:
        The working directory when pytest was invoked.
    """
    rootdir = None
    dirs = get_dirs_from_args(args)
    if inifile:
        inipath_ = absolutepath(inifile)
        inipath: Optional[Path] = inipath_
        inicfg = load_config_dict_from_file(inipath_) or {}
        if rootdir_cmd_arg is None:
            rootdir = inipath_.parent
    else:
        ancestor = get_common_ancestor(invocation_dir, dirs)
        rootdir, inipath, inicfg = locate_config(invocation_dir, [ancestor])
        if rootdir is None and rootdir_cmd_arg is None:
            for possible_rootdir in (ancestor, *ancestor.parents):
                if (possible_rootdir / 'setup.py').is_file():
                    rootdir = possible_rootdir
                    break
            else:
                if dirs != [ancestor]:
                    rootdir, inipath, inicfg = locate_config(invocation_dir, dirs)
                if rootdir is None:
                    rootdir = get_common_ancestor(invocation_dir, [invocation_dir, ancestor])
                    if is_fs_root(rootdir):
                        rootdir = ancestor
    if rootdir_cmd_arg:
        rootdir = absolutepath(os.path.expandvars(rootdir_cmd_arg))
        if not rootdir.is_dir():
            raise UsageError(f"Directory '{rootdir}' not found. Check your '--rootdir' option.")
    assert rootdir is not None
    return (rootdir, inipath, inicfg or {})