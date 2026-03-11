import os
import sys
from pathlib import Path

# Mimic eidos_mcp run_server.sh setup
FORGE_ROOT = Path("/data/data/com.termux/files/home/eidosian_forge").resolve()
sys.path.insert(0, str(FORGE_ROOT / "eidos_mcp" / "src"))
sys.path.insert(0, str(FORGE_ROOT / "agent_forge" / "src"))
sys.path.insert(0, str(FORGE_ROOT / "knowledge_forge" / "src"))
sys.path.insert(0, str(FORGE_ROOT / "crawl_forge" / "src"))
sys.path.insert(0, str(FORGE_ROOT / "lib"))

from crawl_forge.tika_extractor import TikaExtractor, TikaKnowledgeIngester
from knowledge_forge.core.graph import KnowledgeForge

KB_PATH = FORGE_ROOT / "data" / "kb.json"

def debug_ingestion():
    print(f"Checking KB_PATH: {KB_PATH} (exists: {KB_PATH.exists()})")
    
    tika = TikaExtractor()
    print("TikaExtractor initialized.")
    
    try:
        kf = KnowledgeForge(persistence_path=KB_PATH)
        print(f"KnowledgeForge initialized. Node count: {len(kf.nodes)}")
    except Exception as e:
        print(f"KnowledgeForge initialization FAILED: {e}")
        return

    ingester = TikaKnowledgeIngester(tika=tika, knowledge_forge=kf)
    print("TikaKnowledgeIngester initialized.")
    
    test_file = FORGE_ROOT / "archive_forge/research_project/titan_phase1.md"
    print(f"Testing ingestion of: {test_file}")
    
    result = ingester.ingest_file(test_file, tags=["debug"])
    print(f"Ingestion result: {result}")

if __name__ == "__main__":
    debug_ingestion()
