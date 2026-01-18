import unittest
from eidosian_forge.version_forge import Version, VersionForge

class TestVersion(unittest.TestCase):
    def test_parsing(self):
        v = Version("1.2.3-alpha.1+build.123")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)
        self.assertEqual(v.prerelease, "alpha.1")
        self.assertEqual(v.build, "build.123")

    def test_comparison(self):
        self.assertTrue(Version("1.0.0") < Version("2.0.0"))
        self.assertTrue(Version("1.1.0") > Version("1.0.0"))
        self.assertTrue(Version("1.0.1") > Version("1.0.0"))
        self.assertEqual(Version("1.0.0"), Version("1.0.0"))
        # Prerelease
        self.assertTrue(Version("1.0.0-alpha") < Version("1.0.0"))

class TestVersionForge(unittest.TestCase):
    def setUp(self):
        self.forge = VersionForge()

    def test_registration(self):
        self.forge.register_component("core", "1.5.0")
        self.assertEqual(self.forge.get_version("core"), Version("1.5.0"))

    def test_compatibility_caret(self):
        self.forge.register_component("dep", "1.2.3")
        # ^1.2.0 means >=1.2.0 and <2.0.0
        self.assertTrue(self.forge.check_compatibility("app", "dep", "^1.2.0"))
        self.forge.register_component("dep", "1.9.9")
        self.assertTrue(self.forge.check_compatibility("app", "dep", "^1.2.0"))
        self.forge.register_component("dep", "2.0.0")
        self.assertFalse(self.forge.check_compatibility("app", "dep", "^1.2.0"))

    def test_compatibility_tilde(self):
        self.forge.register_component("dep", "1.2.3")
        # ~1.2.0 means >=1.2.0 and <1.3.0
        self.assertTrue(self.forge.check_compatibility("app", "dep", "~1.2.0"))
        self.forge.register_component("dep", "1.2.9")
        self.assertTrue(self.forge.check_compatibility("app", "dep", "~1.2.0"))
        self.forge.register_component("dep", "1.3.0")
        self.assertFalse(self.forge.check_compatibility("app", "dep", "~1.2.0"))

    def test_system_validation(self):
        self.forge.register_component("core", "1.0.0")
        self.forge.register_component("plugin", "1.0.0")
        
        self.forge.add_dependency("plugin", "core", "^1.0.0")
        errors = self.forge.validate_system()
        self.assertEqual(len(errors), 0)
        
        # Introduce break
        self.forge.register_component("core", "2.0.0")
        errors = self.forge.validate_system()
        self.assertEqual(len(errors), 1)
        self.assertIn("plugin requires core ^1.0.0", errors[0])

if __name__ == "__main__":
    unittest.main()
