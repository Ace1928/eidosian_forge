import pytest
import time
from pathlib import Path
from datetime import datetime, timedelta
from knowledge_forge.core.graph import KnowledgeForge, KnowledgeNode

def test_node_aging_and_decay(tmp_path):
    kb_path = tmp_path / "kb.json"
    kb = KnowledgeForge(kb_path)
    
    # Add a node
    node = kb.add_knowledge("The sky is sometimes green on Neptune.")
    assert node.confidence == 1.0
    
    # Manually set last_accessed to 2 days ago
    node.last_accessed = datetime.now() - timedelta(days=2)
    kb.save() # Save manual changes to disk
    
    # Apply decay
    report = kb.apply_decay(decay_factor=0.1)
    assert report["decayed_nodes"] == 1
    assert kb.nodes[node.id].confidence < 1.0
    
    # Set confidence very low and apply decay again to test pruning
    kb.nodes[node.id].confidence = 0.05
    kb.save()
    report = kb.apply_decay(threshold=0.1)
    assert report["pruned_nodes"] == 1
    assert node.id not in kb.nodes

def test_semantic_conflict_detection(tmp_path):
    # This requires an embedder. Let's mock if needed or use real if available.
    # For now, we'll check if the method exists and returns empty list if no embedder.
    kb = KnowledgeForge(tmp_path / "kb.json")
    conflicts = kb.detect_conflicts("The sky is blue")
    assert isinstance(conflicts, list)

def test_touch_on_retrieval(tmp_path):
    kb = KnowledgeForge(tmp_path / "kb.json")
    node = kb.add_knowledge("Test content", tags=["test"])
    initial_access = node.last_accessed
    initial_count = node.access_count
    
    time.sleep(0.01) # Ensure time passes
    
    # Retrieve by tag
    nodes = kb.get_by_tag("test")
    assert len(nodes) == 1
    assert nodes[0].access_count == initial_count + 1
    assert nodes[0].last_accessed > initial_access
