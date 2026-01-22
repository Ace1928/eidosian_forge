import unittest
import gyp.MSVSSettings as MSVSSettings
from io import StringIO
def testConvertToMSBuildSettings_empty(self):
    """Tests an empty conversion."""
    msvs_settings = {}
    expected_msbuild_settings = {}
    actual_msbuild_settings = MSVSSettings.ConvertToMSBuildSettings(msvs_settings, self.stderr)
    self.assertEqual(expected_msbuild_settings, actual_msbuild_settings)
    self._ExpectedWarnings([])