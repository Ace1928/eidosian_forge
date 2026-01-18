#!/usr/bin/env python3
"""
Syncs context/index.json entries to the KnowledgeForge graph.
Creates nodes for filesystem entries and links them.
Uses strict typing and robust error handling.
"""

import json
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Any

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("index_to_kb_sync")

# Paths
ROOT_DIR = Path(__file__).resolve().parents[1]
CONTEXT_INDEX_PATH = ROOT_DIR / "context" / "index.json"
KB_DATA_PATH = ROOT_DIR / "eidosian_forge" / "kb_data.json" # Default persistence

# Inject eidosian_forge into sys.path
EIDOSIAN_FORGE_DIR = ROOT_DIR / "eidosian_forge"
if str(EIDOSIAN_FORGE_DIR) not in sys.path:
    sys.path.insert(0, str(EIDOSIAN_FORGE_DIR))

# Import KnowledgeForge
try:
    # We need to be careful with imports if knowledge_forge is inside eidosian_forge
    # and sys.path includes eidosian_forge.
    # The structure seems to be eidosian_forge/knowledge_forge/knowledge_core.py
    from knowledge_forge.knowledge_core import KnowledgeForge
except ImportError as e:
    logger.error(f"Failed to import KnowledgeForge: {e}")
    # Try alternate import path
    sys.path.append(str(ROOT_DIR))
    try:
        from eidosian_forge.knowledge_forge.knowledge_core import KnowledgeForge
    except ImportError as e2:
        logger.critical(f"Critical Import Error: {e2}")
        sys.exit(1)

def load_index() -> Dict[str, Any]:
    if not CONTEXT_INDEX_PATH.exists():
        logger.error(f"Context index missing at {CONTEXT_INDEX_PATH}")
        sys.exit(1)
    try:
        return json.loads(CONTEXT_INDEX_PATH.read_text())
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in index: {e}")
        sys.exit(1)

def sync_index_to_kb():
    logger.info(f"Loading Knowledge Graph from {KB_DATA_PATH}...")
    # Ensure parent dir exists
    KB_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    kb = KnowledgeForge(persistence_path=KB_DATA_PATH)
    logger.info(f"Loaded {len(kb.nodes)} existing nodes.")

    index_data = load_index()
    directories = index_data.get("directories", [])
    
    if not directories:
        logger.warning("No directories found in index.")
        return

    # Map existing paths to Node IDs to avoid duplicates
    # Heuristic: 'content' field typically holds the path for file nodes
    path_to_nid = {}
    for nid, node in kb.nodes.items():
        if isinstance(node.content, str) and node.content.startswith("/"):
            path_to_nid[node.content] = nid

    stats = {"new": 0, "updated": 0, "skipped": 0, "links": 0}

    # 1. Create/Update Nodes
    for entry in directories:
        path = entry.get("absolute_path")
        if not path:
            continue
            
        is_new = path not in path_to_nid
        
        # Tags
        tags = set(entry.get("section_names", []))
        tags.add("filesystem")
        tags.add(entry.get("type", "other"))
        if entry.get("manual", {}).get("tags"):
            tags.update(entry.get("manual", {}).get("tags"))
            
        # Metadata
        metadata = {
            "name": entry.get("name"),
            "relative_path": entry.get("relative_path"),
            "source": "context_index_sync",
            "last_seen": index_data.get("generated_at"),
            "size_bytes": entry.get("statistics", {}).get("size_bytes"),
            "last_modified": entry.get("statistics", {}).get("last_modified")
        }
        
        # Concepts (naive extraction from path)
        concepts = [p for p in Path(path).parts if p not in ('/', 'home', 'lloyd', '.', '..')]
        
        if is_new:
            node = kb.add_knowledge(
                content=path,
                tags=list(tags),
                concepts=concepts,
                metadata=metadata
            )
            path_to_nid[path] = node.id
            stats["new"] += 1
        else:
            nid = path_to_nid[path]
            node = kb.nodes[nid]
            # Merge tags
            if not tags.issubset(node.tags):
                node.tags.update(tags)
                stats["updated"] += 1
            else:
                stats["skipped"] += 1
            # Update metadata if needed (optional)
            # node.metadata.update(metadata)

    # 2. Create Links (Parent -> Child)
    # We iterate known paths and check if their parent is also known
    for path, nid in path_to_nid.items():
        try:
            parent_path = str(Path(path).parent)
            if parent_path in path_to_nid and parent_path != path:
                parent_nid = path_to_nid[parent_path]
                
                # Check if already linked (KnowledgeForge doesn't expose strict check easily, 
                # but link_nodes handles it via set)
                # We assume parent 'contains' child
                kb.link_nodes(parent_nid, nid)
                stats["links"] += 1
        except Exception as e:
            logger.warning(f"Error linking {path}: {e}")

    # Save
    kb.save()
    logger.info(f"Sync Complete. Stats: {stats}")
    logger.info(f"Total Nodes: {len(kb.nodes)}")

if __name__ == "__main__":
    sync_index_to_kb()
