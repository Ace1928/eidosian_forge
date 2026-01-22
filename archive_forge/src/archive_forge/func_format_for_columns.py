import json
import logging
from optparse import Values
from typing import TYPE_CHECKING, Generator, List, Optional, Sequence, Tuple, cast
from pip._vendor.packaging.utils import canonicalize_name
from pip._internal.cli import cmdoptions
from pip._internal.cli.req_command import IndexGroupCommand
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.exceptions import CommandError
from pip._internal.index.collector import LinkCollector
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_environment
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.network.session import PipSession
from pip._internal.utils.compat import stdlib_pkgs
from pip._internal.utils.misc import tabulate, write_output
def format_for_columns(pkgs: '_ProcessedDists', options: Values) -> Tuple[List[List[str]], List[str]]:
    """
    Convert the package data into something usable
    by output_package_listing_columns.
    """
    header = ['Package', 'Version']
    running_outdated = options.outdated
    if running_outdated:
        header.extend(['Latest', 'Type'])
    has_editables = any((x.editable for x in pkgs))
    if has_editables:
        header.append('Editable project location')
    if options.verbose >= 1:
        header.append('Location')
    if options.verbose >= 1:
        header.append('Installer')
    data = []
    for proj in pkgs:
        row = [proj.raw_name, str(proj.version)]
        if running_outdated:
            row.append(str(proj.latest_version))
            row.append(proj.latest_filetype)
        if has_editables:
            row.append(proj.editable_project_location or '')
        if options.verbose >= 1:
            row.append(proj.location or '')
        if options.verbose >= 1:
            row.append(proj.installer)
        data.append(row)
    return (data, header)