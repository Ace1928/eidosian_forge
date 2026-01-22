import unittest
import gyp.MSVSSettings as MSVSSettings
from io import StringIO
def testConvertToMSBuildSettings_minimal(self):
    """Tests a minimal conversion."""
    msvs_settings = {'VCCLCompilerTool': {'AdditionalIncludeDirectories': 'dir1', 'AdditionalOptions': '/foo', 'BasicRuntimeChecks': '0'}, 'VCLinkerTool': {'LinkTimeCodeGeneration': '1', 'ErrorReporting': '1', 'DataExecutionPrevention': '2'}}
    expected_msbuild_settings = {'ClCompile': {'AdditionalIncludeDirectories': 'dir1', 'AdditionalOptions': '/foo', 'BasicRuntimeChecks': 'Default'}, 'Link': {'LinkTimeCodeGeneration': 'UseLinkTimeCodeGeneration', 'LinkErrorReporting': 'PromptImmediately', 'DataExecutionPrevention': 'true'}}
    actual_msbuild_settings = MSVSSettings.ConvertToMSBuildSettings(msvs_settings, self.stderr)
    self.assertEqual(expected_msbuild_settings, actual_msbuild_settings)
    self._ExpectedWarnings([])