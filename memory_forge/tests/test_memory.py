import unittest
from pathlib import Path
from eidosian_forge.memory_forge import MemoryForge

class TestMemoryForge(unittest.TestCase):
    def setUp(self):
        self.persistence_file = Path("test_memory_persist.json")
        if self.persistence_file.exists():
            self.persistence_file.unlink()
        self.memory = MemoryForge(persistence_path=self.persistence_file)

    def tearDown(self):
        if self.persistence_file.exists():
            self.persistence_file.unlink()

    def test_persistence(self):
        self.memory.remember("Event to persist")
        self.assertTrue(self.persistence_file.exists())
        
        # Load in new instance
        new_memory = MemoryForge(persistence_path=self.persistence_file)
        recent = new_memory.episodic.get_recent(1)
        self.assertEqual(recent[0].content, "Event to persist")

    def test_basic_consolidation(self):
        self.memory.remember("Action 1")
        self.memory.remember("Action 2")
        self.memory.consolidate()
        
        recent = self.memory.episodic.get_recent(2)
        self.assertTrue(recent[0].metadata.get("consolidated"))

    def test_episodic_recording(self):
        self.memory.remember("Learned how to use GIS")
        recent = self.memory.episodic.get_recent(1)
        self.assertEqual(len(recent), 1)
        self.assertEqual(recent[0].content, "Learned how to use GIS")

    def test_semantic_storage(self):
        self.memory.remember("Python is a language", is_fact=True, key="python_info")
        fact = self.memory.semantic.get_fact("python_info")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.content, "Python is a language")

    def test_retrieval(self):
        self.memory.remember("Specific event 123")
        self.memory.remember("Fact about stars", is_fact=True, key="stars")
        
        results = self.memory.retrieve("event")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Specific event 123")
        
        results = self.memory.retrieve("stars")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "Fact about stars")

if __name__ == "__main__":
    unittest.main()
