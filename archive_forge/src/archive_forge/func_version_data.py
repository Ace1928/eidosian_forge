import copy
import re
import urllib
import os_service_types
from keystoneauth1 import _utils as utils
from keystoneauth1 import exceptions
def version_data(self, reverse=False, **kwargs):
    """Get normalized version data.

        Return version data in a structured way.

        :param bool reverse: Reverse the list. reverse=true will mean the
                             returned list is sorted from newest to oldest
                             version.
        :returns: A list of :class:`VersionData` sorted by version number.
        :rtype: list(VersionData)
        """
    data = self.raw_version_data(**kwargs)
    versions = []
    for v in data:
        try:
            version_str = v['id']
        except KeyError:
            _LOGGER.info('Skipping invalid version data. Missing ID.')
            continue
        try:
            links = v['links']
        except KeyError:
            _LOGGER.info('Skipping invalid version data. Missing links')
            continue
        version_number = normalize_version_number(version_str)
        min_microversion = v.get('min_version') or None
        if min_microversion:
            min_microversion = normalize_version_number(min_microversion)
        max_microversion = v.get('max_version')
        if not max_microversion:
            max_microversion = v.get('version') or None
        if max_microversion:
            max_microversion = normalize_version_number(max_microversion)
        next_min_version = v.get('next_min_version') or None
        if next_min_version:
            next_min_version = normalize_version_number(next_min_version)
        not_before = v.get('not_before') or None
        self_url = None
        collection_url = None
        for link in links:
            try:
                rel = link['rel']
                url = _combine_relative_url(self._url, link['href'])
            except (KeyError, TypeError):
                _LOGGER.info('Skipping invalid version link. Missing link URL or relationship.')
                continue
            if rel.lower() == 'self':
                self_url = url
            elif rel.lower() == 'collection':
                collection_url = url
        if not self_url:
            _LOGGER.info('Skipping invalid version data. Missing link to endpoint.')
            continue
        versions.append(VersionData(version=version_number, url=self_url, collection=collection_url, min_microversion=min_microversion, max_microversion=max_microversion, next_min_version=next_min_version, not_before=not_before, status=Status.normalize(v['status']), raw_status=v['status']))
    versions.sort(key=lambda v: v['version'], reverse=reverse)
    return versions