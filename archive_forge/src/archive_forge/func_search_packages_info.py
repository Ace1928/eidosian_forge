import logging
from optparse import Values
from typing import Generator, Iterable, Iterator, List, NamedTuple, Optional
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import ERROR, SUCCESS
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.utils.misc import write_output
def search_packages_info(query: List[str]) -> Generator[_PackageInfo, None, None]:
    """
    Gather details from installed distributions. Print distribution name,
    version, location, and installed files. Installed files requires a
    pip generated 'installed-files.txt' in the distributions '.egg-info'
    directory.
    """
    env = get_default_environment()
    installed = {dist.canonical_name: dist for dist in env.iter_all_distributions()}
    query_names = [canonicalize_name(name) for name in query]
    missing = sorted([name for name, pkg in zip(query, query_names) if pkg not in installed])
    if missing:
        logger.warning('Package(s) not found: %s', ', '.join(missing))

    def _get_requiring_packages(current_dist: BaseDistribution) -> Iterator[str]:
        return (dist.metadata['Name'] or 'UNKNOWN' for dist in installed.values() if current_dist.canonical_name in {canonicalize_name(d.name) for d in dist.iter_dependencies()})
    for query_name in query_names:
        try:
            dist = installed[query_name]
        except KeyError:
            continue
        requires = sorted((req.name for req in dist.iter_dependencies()), key=str.lower)
        required_by = sorted(_get_requiring_packages(dist), key=str.lower)
        try:
            entry_points_text = dist.read_text('entry_points.txt')
            entry_points = entry_points_text.splitlines(keepends=False)
        except FileNotFoundError:
            entry_points = []
        files_iter = dist.iter_declared_entries()
        if files_iter is None:
            files: Optional[List[str]] = None
        else:
            files = sorted(files_iter)
        metadata = dist.metadata
        yield _PackageInfo(name=dist.raw_name, version=str(dist.version), location=dist.location or '', editable_project_location=dist.editable_project_location, requires=requires, required_by=required_by, installer=dist.installer, metadata_version=dist.metadata_version or '', classifiers=metadata.get_all('Classifier', []), summary=metadata.get('Summary', ''), homepage=metadata.get('Home-page', ''), project_urls=metadata.get_all('Project-URL', []), author=metadata.get('Author', ''), author_email=metadata.get('Author-email', ''), license=metadata.get('License', ''), entry_points=entry_points, files=files)