from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def test_unavailable_representation(self):
    error = versionedfile.UnavailableRepresentation(('key',), 'mpdiff', 'fulltext')
    self.assertEqualDiff("The encoding 'mpdiff' is not available for key ('key',) which is encoded as 'fulltext'.", str(error))