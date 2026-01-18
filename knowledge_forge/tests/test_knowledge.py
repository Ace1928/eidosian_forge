import unittest
from pathlib import Path
from eidosian_forge.knowledge_forge import KnowledgeForge

class TestKnowledgeForge(unittest.TestCase):
    def setUp(self):
        self.persistence_file = Path("test_kb_persist.json")
        if self.persistence_file.exists():
            self.persistence_file.unlink()
        self.kb = KnowledgeForge(persistence_path=self.persistence_file)

    def tearDown(self):
        if self.persistence_file.exists():
            self.persistence_file.unlink()

    def test_persistence(self):
        node = self.kb.add_knowledge("Permanent fact")
        self.assertTrue(self.persistence_file.exists())
        
        # Load in new instance
        new_kb = KnowledgeForge(persistence_path=self.persistence_file)
        self.assertIn(node.id, new_kb.nodes)

    def test_tags(self):
        self.kb.add_knowledge("Tagged node", tags=["experimental", "v1"])
        tagged = self.kb.get_by_tag("experimental")
        self.assertEqual(len(tagged), 1)

    def test_pathfinding(self):
        a = self.kb.add_knowledge("A")
        b = self.kb.add_knowledge("B")
        c = self.kb.add_knowledge("C")
        
        self.kb.link_nodes(a.id, b.id)
        self.kb.link_nodes(b.id, c.id)
        
        path = self.kb.find_path(a.id, c.id)
        self.assertEqual(path, [a.id, b.id, c.id])

    def test_add_and_concept_mapping(self):
        node = self.kb.add_knowledge("Python is versatile", concepts=["programming", "python"])
        self.assertIn(node.id, self.kb.concept_map["python"])
        self.assertEqual(self.kb.get_by_concept("python")[0].content, "Python is versatile")

    def test_linking(self):
        node_a = self.kb.add_knowledge("A")
        node_b = self.kb.add_knowledge("B")
        self.kb.link_nodes(node_a.id, node_b.id)
        
        related = self.kb.get_related_nodes(node_a.id)
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].id, node_b.id)

    def test_search(self):
        self.kb.add_knowledge("The sky is blue")
        self.kb.add_knowledge("The grass is green")
        
        results = self.kb.search("sky")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].content, "The sky is blue")

if __name__ == "__main__":
    unittest.main()
