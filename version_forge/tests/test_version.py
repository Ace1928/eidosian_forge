"""Tests for version_forge module."""
import unittest
from version_forge.core.version import SimpleVersion, parse_version
from version_forge.operations.compare import is_compatible, calculate_delta
from version_forge.compatibility.matrix import CompatibilityMatrix


class TestSimpleVersion(unittest.TestCase):
    """Tests for SimpleVersion parsing and comparison."""
    
    def test_parsing(self):
        """Test version string parsing."""
        v = parse_version("1.2.3")
        self.assertEqual(v.major, 1)
        self.assertEqual(v.minor, 2)
        self.assertEqual(v.patch, 3)

    def test_prerelease_parsing(self):
        """Test prerelease version parsing."""
        v = SimpleVersion("1.2.3-alpha.1")
        self.assertEqual(v.prerelease, "alpha.1")

    def test_comparison(self):
        """Test version comparison operators."""
        self.assertTrue(SimpleVersion("1.0.0") < SimpleVersion("2.0.0"))
        self.assertTrue(SimpleVersion("1.1.0") > SimpleVersion("1.0.0"))
        self.assertTrue(SimpleVersion("1.0.1") > SimpleVersion("1.0.0"))
        self.assertEqual(SimpleVersion("1.0.0"), SimpleVersion("1.0.0"))

    def test_prerelease_comparison(self):
        """Prerelease versions are less than release versions."""
        self.assertTrue(SimpleVersion("1.0.0-alpha") < SimpleVersion("1.0.0"))
        self.assertTrue(SimpleVersion("1.0.0-alpha") < SimpleVersion("1.0.0-beta"))


class TestCompatibility(unittest.TestCase):
    """Tests for version compatibility checking."""

    def test_is_compatible_exact(self):
        """Test exact version compatibility."""
        self.assertTrue(is_compatible("1.2.3", "1.2.3"))
        self.assertFalse(is_compatible("1.2.3", "1.2.4"))

    def test_calculate_delta(self):
        """Test version delta calculation."""
        delta = calculate_delta("1.0.0", "2.0.0")
        self.assertEqual(delta["major"], 1)
        self.assertTrue(delta["is_upgrade"])
        
        delta = calculate_delta("1.0.0", "1.1.0")
        self.assertEqual(delta["minor"], 1)
        self.assertEqual(delta["major"], 0)
        
        delta = calculate_delta("1.0.0", "1.0.1")
        self.assertEqual(delta["patch"], 1)


class TestCompatibilityMatrix(unittest.TestCase):
    """Tests for CompatibilityMatrix."""

    def test_matrix_creation(self):
        """Test matrix can be created."""
        matrix = CompatibilityMatrix()
        self.assertIsNotNone(matrix)


if __name__ == "__main__":
    unittest.main()
