import itertools
import operator
import sys
def to_dev(self, dev_count):
    """Return a development version of this semver.

        :param dev_count: The number of commits since the last release.
        """
    return SemanticVersion(self._major, self._minor, self._patch, self._prerelease_type, self._prerelease, dev_count=dev_count)