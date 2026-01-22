import base64
import re
from urllib.parse import urlparse, urlunparse
from ... import bedding, branch, errors, osutils, trace, transport
from ...i18n import gettext
from launchpadlib.credentials import (AccessToken, Credentials,
from launchpadlib.launchpad import Launchpad
Find the revision tree for the LCA of this branch and other.

        :param other: Another LaunchpadBranch
        :return: The RevisionTree of the LCA of this branch and other.
        