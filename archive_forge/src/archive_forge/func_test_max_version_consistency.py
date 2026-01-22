from unittest import mock
import oslotest.base as base
from osc_placement import version
def test_max_version_consistency(self):

    def _convert_to_tuple(str):
        return tuple(map(int, str.split('.')))
    versions = [_convert_to_tuple(ver) for ver in version.SUPPORTED_MICROVERSIONS]
    max_ver = _convert_to_tuple(version.MAX_VERSION_NO_GAP)
    there_is_gap = False
    for i in range(len(versions) - 1):
        j = i + 1
        if versions[j][1] - versions[i][1] != 1:
            there_is_gap = True
            self.assertEqual(max_ver, versions[i])
            break
    if not there_is_gap:
        self.assertEqual(max_ver, versions[-1])