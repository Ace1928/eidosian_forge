import unittest
import os
import json
from pathlib import Path
from eidosian_forge.gis_forge import GisCore

class TestGisCore(unittest.TestCase):
    def setUp(self):
        self.persistence_file = Path("test_gis_persist.json")
        if self.persistence_file.exists():
            self.persistence_file.unlink()
        self.gis = GisCore(persistence_path=self.persistence_file)

    def tearDown(self):
        if self.persistence_file.exists():
            self.persistence_file.unlink()

    def test_set_get_basic(self):
        self.gis.set("app.name", "EidosianForge")
        self.assertEqual(self.gis.get("app.name"), "EidosianForge")

    def test_persistence(self):
        self.gis.set("persist.key", "value")
        self.assertTrue(self.persistence_file.exists())
        
        # Create new GIS instance pointing to same file
        new_gis = GisCore(persistence_path=self.persistence_file)
        self.assertEqual(new_gis.get("persist.key"), "value")

    def test_env_override(self):
        os.environ["EIDOS_APP_PORT"] = "8080"
        self.assertEqual(self.gis.get("app.port"), 8080) # Auto JSON parse
        
        os.environ["EIDOS_COMPLEX_DATA"] = '{"a": 1}'
        self.assertEqual(self.gis.get("complex.data")["a"], 1)

    def test_delete(self):
        self.gis.set("to.delete", 123)
        self.assertTrue(self.gis.delete("to.delete"))
        self.assertIsNone(self.gis.get("to.delete"))

    def test_flatten(self):
        self.gis.update({"a": {"b": 1, "c": {"d": 2}}})
        flat = self.gis.flatten()
        self.assertEqual(flat["a.b"], 1)
        self.assertEqual(flat["a.c.d"], 2)

    def test_subscription(self):
        changes = []
        def callback(key, value):
            changes.append((key, value))
            
        self.gis.subscribe("app", callback)
        self.gis.set("app.status", "starting")
        self.assertEqual(len(changes), 1)

if __name__ == "__main__":
    unittest.main()
