from __future__ import annotations
import json
import os
import requests
import logging
from pathlib import Path
from typing import List, Dict, Any
import sys

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ueo_migration")

# Add gis_forge to path
FORGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FORGE_ROOT / "gis_forge/src"))

from gis_forge.ontology_engine import OntologyEngine, Node

# Configuration
KB_FILE = FORGE_ROOT / "data/kb.json"
MEMORY_FILE = FORGE_ROOT / "data/semantic_memory.json"
EMBEDDING_URL = "http://127.0.0.1:8940/api/embeddings"
EMBEDDING_MODEL = "nomic-embed-text"
ONTOLOGY_DIR = FORGE_ROOT / "data/ontology_db"

engine = OntologyEngine(ONTOLOGY_DIR, dim=768)

def get_embedding(text: str) -> List[float]:
    """Fetch 768-dim embedding from local Ollama service."""
    payload = {
        "model": EMBEDDING_MODEL,
        "prompt": text
    }
    try:
        response = requests.post(EMBEDDING_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        return []

def migrate_kb():
    """Migrate kb.json to UEO."""
    if not KB_FILE.exists():
        logger.warning(f"KB file {KB_FILE} not found.")
        return

    logger.info("Migrating Knowledge Base...")
    with open(KB_FILE, 'r') as f:
        data = json.load(f)
    
    nodes = data.get("nodes", {})
    count = 0
    for node_id, node_data in nodes.items():
        content = node_data.get("content", "")
        if not content: continue
        
        vector = get_embedding(content)
        if not vector: continue
        
        node = Node(
            id=node_id,
            type="fact",
            content=content,
            metadata=node_data.get("metadata", {})
        )
        engine.add_node(node, vector)
        
        # Restore links
        links = node_data.get("links", [])
        for target in links:
            engine.link_nodes(node_id, target, "relates_to")
            
        count += 1
        if count % 10 == 0:
            logger.info(f"Migrated {count} KB nodes...")
            
    logger.info(f"Successfully migrated {count} KB nodes.")

def migrate_memory():
    """Migrate semantic_memory.json to UEO."""
    if not MEMORY_FILE.exists():
        logger.warning(f"Memory file {MEMORY_FILE} not found.")
        return

    logger.info("Migrating Semantic Memory...")
    with open(MEMORY_FILE, 'r') as f:
        data = json.load(f)
    
    count = 0
    for mem_id, mem_data in data.items():
        content = mem_data.get("content", "")
        if not content: continue
        
        # ALWAYS re-embed to ensure 768-dim parity
        vector = get_embedding(content)
        if not vector: continue
        
        node = Node(
            id=mem_id,
            type="memory",
            content=content,
            metadata={
                "created_at": mem_data.get("created_at"),
                "importance": mem_data.get("importance"),
                **mem_data.get("metadata", {})
            }
        )
        engine.add_node(node, vector)
        count += 1
        if count % 10 == 0:
            logger.info(f"Migrated {count} memory nodes...")
            
    logger.info(f"Successfully migrated {count} memory nodes.")

if __name__ == "__main__":
    migrate_kb()
    migrate_memory()
    logger.info("UEO Migration Complete.")
