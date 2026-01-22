from typing import Optional
from pip._internal.models.direct_url import ArchiveInfo, DirectUrl, DirInfo, VcsInfo
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import vcs
def direct_url_as_pep440_direct_reference(direct_url: DirectUrl, name: str) -> str:
    """Convert a DirectUrl to a pip requirement string."""
    direct_url.validate()
    requirement = name + ' @ '
    fragments = []
    if isinstance(direct_url.info, VcsInfo):
        requirement += '{}+{}@{}'.format(direct_url.info.vcs, direct_url.url, direct_url.info.commit_id)
    elif isinstance(direct_url.info, ArchiveInfo):
        requirement += direct_url.url
        if direct_url.info.hash:
            fragments.append(direct_url.info.hash)
    else:
        assert isinstance(direct_url.info, DirInfo)
        requirement += direct_url.url
    if direct_url.subdirectory:
        fragments.append('subdirectory=' + direct_url.subdirectory)
    if fragments:
        requirement += '#' + '&'.join(fragments)
    return requirement