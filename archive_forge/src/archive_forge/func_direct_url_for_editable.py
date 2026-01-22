from typing import Optional
from pip._internal.models.direct_url import ArchiveInfo, DirectUrl, DirInfo, VcsInfo
from pip._internal.models.link import Link
from pip._internal.utils.urls import path_to_url
from pip._internal.vcs import vcs
def direct_url_for_editable(source_dir: str) -> DirectUrl:
    return DirectUrl(url=path_to_url(source_dir), info=DirInfo(editable=True))