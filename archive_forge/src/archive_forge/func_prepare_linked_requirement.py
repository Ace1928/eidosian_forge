import mimetypes
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.distributions.installed import InstalledDistribution
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_metadata_distribution
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs
def prepare_linked_requirement(self, req: InstallRequirement, parallel_builds: bool=False) -> BaseDistribution:
    """Prepare a requirement to be obtained from req.link."""
    assert req.link
    self._log_preparing_link(req)
    with indent_log():
        file_path = None
        if self.download_dir is not None and req.link.is_wheel:
            hashes = self._get_linked_req_hashes(req)
            file_path = _check_download_dir(req.link, self.download_dir, hashes, warn_on_hash_mismatch=not req.is_wheel_from_cache)
        if file_path is not None:
            self._downloaded[req.link.url] = file_path
        else:
            metadata_dist = self._fetch_metadata_only(req)
            if metadata_dist is not None:
                req.needs_more_preparation = True
                return metadata_dist
        return self._prepare_linked_requirement(req, parallel_builds)