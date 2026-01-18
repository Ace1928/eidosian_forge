"""
Tests for the util.classregistry module.

This module tests the ClassRegistry utility for introspecting
classes and functions in the Stratum codebase.
"""

import unittest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from util.classregistry import ClassRegistry


class TestClassRegistry(unittest.TestCase):
    """Tests for the ClassRegistry class."""

    def setUp(self):
        """Set up test fixtures."""
        # Use the local package path
        self.registry = ClassRegistry(root_package='core')

    def test_initialization(self):
        """Test ClassRegistry initialization."""
        self.assertEqual(self.registry.root_package, 'core')
        self.assertIsInstance(self.registry._module_cache, dict)

    def test_list_modules(self):
        """Test listing modules under root package."""
        modules = self.registry.list_modules()
        self.assertIsInstance(modules, list)
        # Should find at least the core modules
        # Note: This may need adjustment based on actual package structure

    def test_load_module(self):
        """Test loading a module by name."""
        module = self.registry.load_module('core.types')
        self.assertIsNotNone(module)

    def test_load_module_caches(self):
        """Test that loaded modules are cached."""
        mod1 = self.registry.load_module('core.types')
        mod2 = self.registry.load_module('core.types')
        self.assertIs(mod1, mod2)

    def test_load_module_invalid(self):
        """Test loading invalid module raises ImportError."""
        with self.assertRaises(ImportError):
            self.registry.load_module('nonexistent.module.path')

    def test_list_classes(self):
        """Test listing classes in a module."""
        classes = self.registry.list_classes('core.types')
        self.assertIsInstance(classes, list)
        self.assertIn('Vec2', classes)
        self.assertIn('Cell', classes)

    def test_list_classes_invalid_module(self):
        """Test listing classes in invalid module returns empty list."""
        classes = self.registry.list_classes('nonexistent.module')
        self.assertEqual(classes, [])

    def test_list_functions(self):
        """Test listing functions in a module."""
        functions = self.registry.list_functions('core.types')
        self.assertIsInstance(functions, list)
        self.assertIn('dot', functions)
        self.assertIn('clamp', functions)

    def test_list_functions_invalid_module(self):
        """Test listing functions in invalid module returns empty list."""
        functions = self.registry.list_functions('nonexistent.module')
        self.assertEqual(functions, [])

    def test_get_source_class(self):
        """Test getting source code for a class."""
        source = self.registry.get_source('core.types', 'Vec2')
        self.assertIsInstance(source, str)
        self.assertIn('class Vec2', source)
        self.assertIn('def __add__', source)

    def test_get_source_function(self):
        """Test getting source code for a function."""
        source = self.registry.get_source('core.types', 'clamp')
        self.assertIsInstance(source, str)
        self.assertIn('def clamp', source)

    def test_get_source_nonexistent_object(self):
        """Test getting source for nonexistent object returns None."""
        source = self.registry.get_source('core.types', 'NonexistentClass')
        self.assertIsNone(source)

    def test_get_source_invalid_module(self):
        """Test getting source from invalid module returns None."""
        source = self.registry.get_source('nonexistent.module', 'SomeClass')
        self.assertIsNone(source)


class TestClassRegistryFabric(unittest.TestCase):
    """Tests for ClassRegistry with fabric module."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ClassRegistry(root_package='core')

    def test_list_classes_fabric(self):
        """Test listing classes in fabric module."""
        classes = self.registry.list_classes('core.fabric')
        self.assertIn('Fabric', classes)
        self.assertIn('Mixture', classes)

    def test_get_source_fabric(self):
        """Test getting Fabric class source."""
        source = self.registry.get_source('core.fabric', 'Fabric')
        self.assertIsInstance(source, str)
        self.assertIn('class Fabric', source)


class TestClassRegistryLedger(unittest.TestCase):
    """Tests for ClassRegistry with ledger module."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ClassRegistry(root_package='core')

    def test_list_classes_ledger(self):
        """Test listing classes in ledger module."""
        classes = self.registry.list_classes('core.ledger')
        self.assertIn('Ledger', classes)
        self.assertIn('EntropySource', classes)
        self.assertIn('EntropyRecord', classes)


class TestClassRegistryConfig(unittest.TestCase):
    """Tests for ClassRegistry with config module."""

    def setUp(self):
        """Set up test fixtures."""
        self.registry = ClassRegistry(root_package='core')

    def test_list_classes_config(self):
        """Test listing classes in config module."""
        classes = self.registry.list_classes('core.config')
        self.assertIn('EngineConfig', classes)


if __name__ == "__main__":
    unittest.main()
