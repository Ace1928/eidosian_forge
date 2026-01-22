from __future__ import absolute_import, division, print_function
import hashlib
import os
import posixpath
import shutil
import io
import tempfile
import traceback
import re
from ansible_collections.community.general.plugins.module_utils.version import LooseVersion
from ansible.module_utils.ansible_release import __version__ as ansible_version
from re import match
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.six.moves.urllib.parse import urlparse
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
def find_uri_for_artifact(self, artifact):
    if artifact.version_by_spec:
        artifact.version = self.find_version_by_spec(artifact)
    if artifact.version == 'latest':
        artifact.version = self.find_latest_version_available(artifact)
    if artifact.is_snapshot():
        if self.local:
            return self._uri_for_artifact(artifact, artifact.version)
        path = '/%s/%s' % (artifact.path(), self.metadata_file_name)
        content = self._getContent(self.base + path, 'Failed to retrieve the maven metadata file: ' + path)
        xml = etree.fromstring(content)
        for snapshotArtifact in xml.xpath('/metadata/versioning/snapshotVersions/snapshotVersion'):
            classifier = snapshotArtifact.xpath('classifier/text()')
            artifact_classifier = classifier[0] if classifier else ''
            extension = snapshotArtifact.xpath('extension/text()')
            artifact_extension = extension[0] if extension else ''
            if artifact_classifier == artifact.classifier and artifact_extension == artifact.extension:
                return self._uri_for_artifact(artifact, snapshotArtifact.xpath('value/text()')[0])
        timestamp_xmlpath = xml.xpath('/metadata/versioning/snapshot/timestamp/text()')
        if timestamp_xmlpath:
            timestamp = timestamp_xmlpath[0]
            build_number = xml.xpath('/metadata/versioning/snapshot/buildNumber/text()')[0]
            return self._uri_for_artifact(artifact, artifact.version.replace('SNAPSHOT', timestamp + '-' + build_number))
    return self._uri_for_artifact(artifact, artifact.version)