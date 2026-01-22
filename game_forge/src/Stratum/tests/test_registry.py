"""
Tests for the core.registry module.

This module tests the SpeciesRegistry class including species creation,
quantization, hashing, and migrations.
"""

import unittest
import sys
import os
import tempfile
import json
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.registry import SpeciesRegistry, Species


class TestSpecies(unittest.TestCase):
    """Tests for the Species dataclass."""

    def test_creation(self):
        """Test Species creation."""
        s = Species(
            id="test123",
            he_props={"HE/rho_max": 0.5},
            le_props={"LE/bond": 0.1},
        )
        self.assertEqual(s.id, "test123")
        self.assertEqual(s.he_props["HE/rho_max"], 0.5)
        self.assertEqual(s.le_props["LE/bond"], 0.1)

    def test_default_values(self):
        """Test Species default values."""
        s = Species(id="test", he_props={}, le_props={})
        self.assertEqual(s.schema_version, 1)
        self.assertEqual(s.provenance, {})
        self.assertEqual(s.stability_stats, {})


class TestSpeciesRegistry(unittest.TestCase):
    """Tests for the SpeciesRegistry class."""

    def setUp(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = ["HE/rho_max", "HE/chi", "HE/eta"]
        self.le_props = ["LE/bond"]

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_initialization_creates_file(self):
        """Test that registry creates file if it doesn't exist."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        self.assertTrue(os.path.exists(self.registry_path))

    def test_initialization_loads_existing(self):
        """Test that registry loads existing file."""
        # Create a registry and save a species
        reg1 = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = reg1.get_or_create_species({"HE/rho_max": 0.5, "HE/chi": 0.3, "HE/eta": 0.2})
        reg1.save()
        # Create new registry instance and verify species is loaded
        reg2 = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        self.assertIn(species.id, reg2.species)

    def test_quantise_props(self):
        """Test property quantization."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        props = {"HE/rho_max": 0.5, "HE/chi": 0.0, "HE/eta": 1.0}
        q = registry.quantise_props(props)
        self.assertEqual(q["HE/chi"], 0)  # 0.0 -> 0
        self.assertEqual(q["HE/eta"], 255)  # 1.0 -> 255
        # 0.5 -> 127 or 128 depending on rounding
        self.assertIn(q["HE/rho_max"], [127, 128])

    def test_quantise_props_clamps(self):
        """Test that quantization clamps values to [0,1]."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        props = {"HE/rho_max": -0.5, "HE/chi": 1.5, "HE/eta": 0.5}
        q = registry.quantise_props(props)
        self.assertEqual(q["HE/rho_max"], 0)  # -0.5 clamped to 0
        self.assertEqual(q["HE/chi"], 255)  # 1.5 clamped to 1.0

    def test_dequantise_props(self):
        """Test property dequantization."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        q = {"HE/rho_max": 128, "HE/chi": 0, "HE/eta": 255}
        props = registry.dequantise_props(q)
        self.assertAlmostEqual(props["HE/chi"], 0.0, places=5)
        self.assertAlmostEqual(props["HE/eta"], 1.0, places=5)

    def test_get_or_create_species_new(self):
        """Test creating a new species."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species(
            {"HE/rho_max": 0.5, "HE/chi": 0.3, "HE/eta": 0.2}
        )
        self.assertIsInstance(species, Species)
        self.assertIn(species.id, registry.species)

    def test_get_or_create_species_existing(self):
        """Test getting an existing species."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        props = {"HE/rho_max": 0.5, "HE/chi": 0.3, "HE/eta": 0.2}
        s1 = registry.get_or_create_species(props)
        s2 = registry.get_or_create_species(props)
        self.assertEqual(s1.id, s2.id)
        # Should be the same object
        self.assertIs(s1, s2)

    def test_get_or_create_species_with_provenance(self):
        """Test creating species with provenance."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species(
            {"HE/rho_max": 0.5, "HE/chi": 0.3, "HE/eta": 0.2},
            provenance={"source": "test", "tick": 10},
        )
        self.assertEqual(species.provenance["source"], "test")
        self.assertEqual(species.provenance["tick"], 10)

    def test_save_and_load(self):
        """Test saving and loading registry."""
        reg1 = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        s = reg1.get_or_create_species({"HE/rho_max": 0.5})
        reg1.save()
        # Load data directly
        with open(self.registry_path, 'r') as f:
            data = json.load(f)
        self.assertIn(s.id, data["species"])

    def test_mark_stable(self):
        """Test marking a species as stable."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species({"HE/rho_max": 0.5})
        registry.mark_stable(species.id, {"stable": True, "iterations": 100})
        self.assertEqual(species.stability_stats["stable"], True)
        self.assertEqual(species.stability_stats["iterations"], 100)

    def test_mark_stable_nonexistent(self):
        """Test marking nonexistent species does nothing."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        # Should not raise
        registry.mark_stable("nonexistent_id", {"stable": True})

    def test_migrate_le_properties(self):
        """Test migrating LE properties."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species({"HE/rho_max": 0.5, "HE/chi": 0.3})
        # Define a migration function
        def compute_new_prop(he_props, species_id):
            return he_props.get("HE/rho_max", 0.0) * 2.0
        registry.migrate_le_properties({"LE/new_prop": compute_new_prop}, new_version=2)
        # Check that the new property was added
        self.assertIn("LE/new_prop", species.le_props)
        self.assertAlmostEqual(species.le_props["LE/new_prop"], 1.0, places=2)
        # Check schema version updated
        self.assertEqual(species.schema_version, 2)
        self.assertEqual(registry.schema_version, 2)

    def test_species_id_deterministic(self):
        """Test that species ID is deterministic from HE props."""
        reg1 = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        props = {"HE/rho_max": 0.5, "HE/chi": 0.3, "HE/eta": 0.1}
        s1 = reg1.get_or_create_species(props)
        # Create a fresh registry
        path2 = os.path.join(self.temp_dir, "registry2.json")
        reg2 = SpeciesRegistry(path2, self.he_props, self.le_props)
        s2 = reg2.get_or_create_species(props)
        self.assertEqual(s1.id, s2.id)

    def test_different_props_different_ids(self):
        """Test that different HE props produce different IDs."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        s1 = registry.get_or_create_species({"HE/rho_max": 0.5})
        s2 = registry.get_or_create_species({"HE/rho_max": 0.6})
        self.assertNotEqual(s1.id, s2.id)


class TestSpeciesRegistryEdgeCases(unittest.TestCase):
    """Edge case tests for SpeciesRegistry."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = os.path.join(self.temp_dir, "registry.json")
        self.he_props = ["HE/rho_max", "HE/chi"]
        self.le_props = []

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_he_props(self):
        """Test creating species with empty HE props dict."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species({})
        self.assertIsInstance(species, Species)

    def test_missing_props_filled_with_zero(self):
        """Test that missing HE props default to zero."""
        registry = SpeciesRegistry(self.registry_path, self.he_props, self.le_props)
        species = registry.get_or_create_species({"HE/rho_max": 0.5})
        self.assertIn("HE/chi", species.he_props)

    def test_nested_directory_creation(self):
        """Test that nested directories are created."""
        nested_path = os.path.join(self.temp_dir, "a", "b", "c", "registry.json")
        registry = SpeciesRegistry(nested_path, self.he_props, self.le_props)
        self.assertTrue(os.path.exists(nested_path))


if __name__ == "__main__":
    unittest.main()
