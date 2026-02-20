#!/usr/bin/env python3
"""
Test suite for the EidosMemorySystem.
"""

import json
import logging
import time
import unittest
from unittest.mock import patch

from memory_system import AgenticMemoryUnit, EidosMemorySystem, RetrievalError, UnitNotFoundError

from tests.test_utils import LLMControllerError, MockLLMController

logger = logging.getLogger(__name__)


class TestEidosMemorySystem(unittest.TestCase):
    """
    Comprehensive test suite for the EidosMemorySystem module.

    Tests all major functionality:
    - CRUD operations (create, read, update, delete)
    - Semantic search functionality
    - Memory unit metadata handling
    - Memory evolution processes
    - Error handling
    - Memory statistics tracking
    - Bulk operations
    """

    @classmethod
    def setUpClass(cls):
        """
        Optionally load expensive resources once for all tests
        (e.g., a shared ChromaDB collection, etc.).
        """
        logging.info("Setting up TestEidosMemorySystem suite...")

    def setUp(self) -> None:
        """Initialize a fresh EidosMemorySystem before each test."""
        # Use a mock LLM controller to avoid real LLM calls during tests
        self.mock_controller = MockLLMController()

        # Set up default mock response for content analysis
        mock_analysis = {
            "keywords": ["test", "memory", "system"],
            "context": "Testing",
            "tags": ["unit_test", "automated", "memory"],
        }
        self.mock_controller.mock_response = json.dumps(mock_analysis)

        # Create the memory system with our mock controller
        self.memory_system = EidosMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="ollama",
            llm_model="deepseek-r1:1.5b",
            llm_controller=self.mock_controller,
        )

    # -------------------------------------------------------------------------
    # Basic CRUD Tests
    # -------------------------------------------------------------------------
    def test_create_memory(self) -> None:
        """Test creating a new memory unit."""
        content = "Test memory content"
        memory_id = self.memory_system.create_unit(content)

        self.assertIsNotNone(memory_id)
        memory = self.memory_system.read_unit(memory_id)
        self.assertIsNotNone(memory)
        self.assertEqual(memory.content, content)

        # Test that system statistics are updated
        self.assertEqual(self.memory_system.total_units_created, 1)

        # Test that auto-analysis was applied
        self.assertGreater(len(memory.keywords), 0)
        self.assertGreater(len(memory.tags), 0)
        self.assertNotEqual(memory.context, "General")

    def test_read_memory_success(self) -> None:
        """Test successfully reading a memory unit."""
        content = "Test memory for reading"
        memory_id = self.memory_system.create_unit(content)

        # Force different timestamps by manipulating the memory unit directly
        original_unit = self.memory_system.units[memory_id]
        original_timestamp = original_unit.timestamp

        # Ensure timestamp and last_accessed are initially the same
        self.assertEqual(original_unit.timestamp, original_unit.last_accessed)

        # Now read it, which should update last_accessed
        memory = self.memory_system.read_unit(memory_id)

        # Verify that timestamp didn't change but last_accessed did
        self.assertEqual(memory.timestamp, original_timestamp)
        self.assertNotEqual(memory.last_accessed, memory.timestamp)

        # Test that access is recorded
        self.assertEqual(memory.retrieval_count, 1)

    def test_read_memory_not_found(self) -> None:
        """Test that reading a non-existent memory raises the correct exception."""
        with self.assertRaises(UnitNotFoundError):
            self.memory_system.read_unit("non_existent_id")

    def test_update_memory_success(self) -> None:
        """Test successfully updating an existing memory unit."""
        original = "Original content"
        memory_id = self.memory_system.create_unit(original)

        # Record the original state for comparison
        original_memory = self.memory_system.read_unit(memory_id)

        # Update with new content
        new_content = "Updated content"
        new_keywords = ["updated", "content"]

        success = self.memory_system.update_unit(memory_id, content=new_content, keywords=new_keywords)
        self.assertTrue(success)

        # Verify the update
        updated_memory = self.memory_system.read_unit(memory_id)
        self.assertEqual(updated_memory.content, new_content)
        self.assertEqual(updated_memory.keywords, new_keywords)

        # Test that evolution history is updated
        self.assertEqual(len(updated_memory.evolution_history), 1)
        self.assertEqual(updated_memory.evolution_history[0]["type"], "manual_update")
        self.assertEqual(updated_memory.evolution_history[0]["details"]["previous"]["content"], original)
        self.assertEqual(updated_memory.evolution_history[0]["details"]["current"]["content"], new_content)

    def test_update_memory_not_found(self) -> None:
        """Test updating a non-existent memory unit."""
        invalid_update = self.memory_system.update_unit("fake_id", content="does not matter")
        self.assertFalse(invalid_update)

    def test_delete_memory_success(self) -> None:
        """Test successfully deleting a memory unit."""
        content = "Memory to delete"
        memory_id = self.memory_system.create_unit(content)

        initial_count = self.memory_system.total_units_created

        # Delete the unit
        success = self.memory_system.delete_unit(memory_id)
        self.assertTrue(success)

        # Verify it's gone
        with self.assertRaises(UnitNotFoundError):
            self.memory_system.read_unit(memory_id)

        # Check that deletion statistics are updated
        self.assertEqual(self.memory_system.total_units_deleted, 1)

        # Check that the total count is consistent
        self.assertEqual(len(self.memory_system.units), initial_count - 1)

    def test_delete_memory_not_found(self) -> None:
        """Test deleting a non-existent memory unit."""
        invalid_delete = self.memory_system.delete_unit("non_existent_id")
        self.assertFalse(invalid_delete)

    # -------------------------------------------------------------------------
    # Search Tests
    # -------------------------------------------------------------------------
    def test_search_memories(self) -> None:
        """Test hybrid semantic search."""
        docs = [
            "Python programming language",
            "JavaScript web development",
            "Python data science",
            "Ruby on Rails framework",
            "Python machine learning",
        ]
        for doc in docs:
            self.memory_system.create_unit(doc)

        with self.subTest("Single-term search"):
            results = self.memory_system.search_units("Python")
            self.assertGreater(len(results), 0)
            # Python-related documents should score higher
            for result in results[:3]:  # Top 3 results
                self.assertIn("Python", result["content"])

        with self.subTest("Multi-term search"):
            results = self.memory_system.search_units("machine learning")
            self.assertGreater(len(results), 0)
            self.assertIn("machine learning", results[0]["content"].lower())

        with self.subTest("Non-existent term"):
            results = self.memory_system.search_units("XYZNonExistentTerm123")
            self.assertEqual(len(results), 0)

        with self.subTest("Empty query"):
            results = self.memory_system.search_units("")
            self.assertEqual(len(results), 0)

        # Check search statistics tracking
        self.assertEqual(self.memory_system.total_searches, 4)

    @patch("retrievers.SimpleEmbeddingRetriever.search")
    @patch("retrievers.ChromaRetriever.search")
    def test_search_error_handling(self, mock_chroma_search, mock_embedding_search) -> None:
        """Test that search failures are handled gracefully."""
        # Set up mocks to raise exceptions
        mock_chroma_search.side_effect = Exception("Chroma search failed")
        mock_embedding_search.side_effect = Exception("Embedding search failed")

        # When both search methods fail, should raise RetrievalError
        with self.assertRaises(RetrievalError):
            self.memory_system.search_units("test")

    # -------------------------------------------------------------------------
    # Metadata Tests
    # -------------------------------------------------------------------------
    def test_memory_metadata(self) -> None:
        """Test that memory metadata is stored and retrieved."""
        test_content = "This is test content for metadata testing."

        # Create memory with metadata
        mem_id = self.memory_system.create_unit(
            test_content, context="Testing", tags=["metadata", "test"], category="Unit Test"
        )

        # Search for this memory
        results = self.memory_system.search_units("metadata testing")

        # Check metadata is correctly retrieved
        self.assertGreater(len(results), 0, "Should find at least one result")
        self.assertEqual(results[0]["context"], "Testing")
        self.assertIn("metadata", results[0]["keywords"])

    # -------------------------------------------------------------------------
    # Evolution Tests
    # -------------------------------------------------------------------------
    def test_memory_evolution_threshold(self) -> None:
        """Test that evolution is triggered at the right threshold."""
        self.memory_system.evolution_threshold = 3
        self.memory_system.evolution_count = 0

        # Create units up to threshold - 1
        docs = [
            "Deep learning neural networks",
            "Neural network architectures",
        ]
        for doc in docs:
            self.memory_system.create_unit(doc)

        # Evolution counter should increase but not trigger
        self.assertEqual(self.memory_system.evolution_count, 2)

        # Add one more to trigger evolution
        with patch.object(self.memory_system, "_process_evolution") as mock_evolve:
            mock_evolve.return_value = True  # Evolution successful
            self.memory_system.create_unit("Training deep neural networks")
            mock_evolve.assert_called_once()

        # Counter should be reset
        self.assertEqual(self.memory_system.evolution_count, 0)

    def test_memory_evolution_comprehensive(self) -> None:
        """Test memory evolution with merging via a mock LLM controller."""
        # Setup mock to simulate LLM deciding to evolve via merge
        mock_evolution_response = json.dumps(
            {
                "should_evolve": True,
                "evolution_type": ["merge"],
                "reasoning": "Test reasoning",
                "affected_units": ["unit1", "unit2"],
                "evolution_details": {
                    "new_context": "Updated Context",
                    "new_keywords": ["keyword1", "keyword2"],
                    "new_relationships": ["rel1", "rel2"],
                },
            }
        )

        # Create test memory units first
        self.memory_system.units["unit1"] = AgenticMemoryUnit(content="Unit 1 content", id="unit1")
        self.memory_system.units["unit2"] = AgenticMemoryUnit(content="Unit 2 content", id="unit2")

        # Reset the evolution history to avoid extra entries
        unit1 = self.memory_system.read_unit("unit1")
        unit1.evolution_history = []
        unit2 = self.memory_system.read_unit("unit2")
        unit2.evolution_history = []

        # Use mock_controller instead of llm_controller
        self.mock_controller.mock_response = mock_evolution_response

        # Create a memory unit that should trigger evolution with our mock
        mem_id = self.memory_system.create_unit(
            "This content should trigger evolution through merging", context="Test Context"
        )

        # Verify the evolution occurred through merge
        with self.assertRaises(UnitNotFoundError):
            self.memory_system.read_unit(mem_id)

        # Check that target unit was updated
        updated_mem = self.memory_system.read_unit("unit1")
        self.assertEqual(len(updated_mem.evolution_history), 1)
        self.assertEqual(updated_mem.evolution_history[0]["type"], "merge_recipient")

    def test_memory_evolution_update(self) -> None:
        """Test memory evolution with updating (not merging)."""
        # Setup mock to simulate LLM deciding to evolve via update
        mock_evolution_response = json.dumps(
            {
                "should_evolve": True,
                "evolution_type": ["update"],
                "reasoning": "Test reasoning",
                "affected_units": [],
                "evolution_details": {
                    "new_context": "Enhanced Context",
                    "new_keywords": ["enhanced", "updated"],
                    "new_relationships": [],
                },
            }
        )

        # Use mock_controller instead of llm_controller
        self.mock_controller.mock_response = mock_evolution_response

        # Create a memory unit that should evolve via update
        mem_id = self.memory_system.create_unit(
            "This content should be updated through evolution",
            context="Original Context",
            evolution_history=[],  # Start with empty history
        )

        # Manually trigger evolution
        unit = self.memory_system.read_unit(mem_id)
        self.memory_system._process_evolution(unit)

        # Verify the update occurred
        updated_mem = self.memory_system.read_unit(mem_id)
        self.assertEqual(len(updated_mem.evolution_history), 1)
        self.assertEqual(updated_mem.context, "Enhanced Context")
        self.assertIn("enhanced", updated_mem.keywords)

    def test_failed_evolution(self) -> None:
        """Test graceful handling of failed evolution attempts."""
        system = EidosMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend="ollama",
            llm_model="deepseek-r1:1.5b",
            evolution_threshold=2,
            llm_controller=self.mock_controller,
        )

        # Create initial units
        id1 = system.create_unit("Test memory 1")
        id2 = system.create_unit("Test memory 2")
        self.assertEqual(system.evolution_count, 2)

        # Set invalid JSON response to trigger failure
        self.mock_controller.mock_response = "invalid json"

        # Create unit that triggers evolution
        id3 = system.create_unit("Test memory 3")

        # Check that failure was handled gracefully
        self.assertIn(id3, system.units)
        self.assertEqual(system.evolution_count, 3)  # Counter still incremented
        self.assertEqual(system.failed_evolutions, 1)  # Failure tracked

    def test_evolution_with_exception(self) -> None:
        """Test graceful handling of exceptions during evolution."""
        # Setup mock to throw an exception
        self.mock_controller.set_errors([LLMControllerError("Test error")])

        # Create memory unit to trigger evolution
        mem_id = self.memory_system.create_unit("This should trigger evolution but encounter an error")

        # Verify memory still exists
        mem = self.memory_system.read_unit(mem_id)
        self.assertEqual(mem.id, mem_id)

        # Force evolution to catch exception
        self.memory_system._process_evolution(mem)

        # Check that the error was properly counted
        self.assertEqual(self.memory_system.failed_evolutions, 1)

    # -------------------------------------------------------------------------
    # Statistics and Analytics Tests
    # -------------------------------------------------------------------------
    def test_memory_statistics(self) -> None:
        """Test memory system statistics tracking."""
        system = EidosMemorySystem(model_name="all-MiniLM-L6-v2", llm_controller=self.mock_controller)

        # Track initial state
        initial_created = system.total_units_created
        initial_time = system.creation_time

        # Create and perform operations
        id1 = system.create_unit("Stats test 1")
        id2 = system.create_unit("Stats test 2")

        system.search_units("test")
        system.search_units("stats")

        system.delete_unit(id1)

        # Check statistics
        self.assertEqual(system.total_units_created, initial_created + 2)
        self.assertEqual(system.total_units_deleted, 1)
        self.assertEqual(system.total_searches, 2)
        self.assertGreaterEqual(time.time(), initial_time)

    # -------------------------------------------------------------------------
    # Error Handling Tests
    # -------------------------------------------------------------------------
    def test_content_analysis_error_handling(self) -> None:
        """Test that content analysis errors are handled gracefully."""
        # Set mock to fail
        self.mock_controller.set_errors([Exception("Analysis failed")])

        # Should still create the unit with default metadata
        memory_id = self.memory_system.create_unit("This should still work")

        # Unit should exist with default metadata
        memory = self.memory_system.read_unit(memory_id)
        self.assertEqual(memory.context, "General")
        self.assertEqual(memory.keywords, [])
        self.assertEqual(memory.tags, [])


if __name__ == "__main__":
    unittest.main()
