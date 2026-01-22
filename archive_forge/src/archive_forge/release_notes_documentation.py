from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.updater import installers
import requests
from six.moves import StringIO
Creates a diff of the release notes between the two versions.

    The release notes are returned in reversed order (most recent first).

    Args:
      start_version: str, The version at which to start the diff.  This should
        be the later of the two versions.  The diff will start with this version
        and go backwards in time until end_version is hit.  If None, the diff
        will start at the most recent entry.
      end_version: str, The version at which to stop the diff.  This should be
        the version you are currently on.  The diff is accumulated until this
        version it hit.  This version is not included in the diff.  If None,
        the diff will include through the end of all release notes.

    Returns:
      [(version, text)], The list of release notes in the diff from most recent
      to least recent.  Each item is a tuple of the version string and the
      release notes text for that version.  Returns None if either of the
      versions are not present in the release notes.
    