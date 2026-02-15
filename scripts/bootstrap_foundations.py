"""
Orchestration script to initialize Eidosian Foundation forges.
"""
import sys
import os
from pathlib import Path

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parent.parent))).resolve()
for extra in (FORGE_DIR / "lib", FORGE_DIR):
    extra_str = str(extra)
    if extra.exists() and extra_str not in sys.path:
        sys.path.insert(0, extra_str)

from eidosian_core import eidosian
from gis_forge import GisCore
from diagnostics_forge import DiagnosticsForge
from type_forge import TypeCore
from version_forge import VersionForge
from memory_forge import MemoryForge
from knowledge_forge import KnowledgeForge
from code_forge import CodeForge
from refactor_forge import RefactorForge
from crawl_forge import CrawlForge
from llm_forge import LLMForge
from figlet_forge import FigletForge
from file_forge import FileForge
from doc_forge import DocForge
from agent_forge import AgentForge

@eidosian()
def bootstrap():
    print("🚀 Initializing Eidosian Foundation...")
    home_dir = FORGE_DIR.parent
    
    # 1. Initialize GIS with persistence
    gis = GisCore(persistence_path=FORGE_DIR / "gis_data.json")
    gis.set("system.version", "3.14.15", persist=False)
    gis.set("system.status", "bootstrapping", persist=False)
    
    # 2. Initialize Diagnostics
    diag = DiagnosticsForge(log_dir=str(FORGE_DIR / "logs"), service_name="foundation_bootstrap")
    diag.log_event("INFO", "Foundation bootstrap started")
    
    # 3. Initialize Version Forge
    versions = VersionForge()
    for forge_name in ["gis", "diagnostics", "type", "version", "memory", "knowledge", "code", "refactor", "crawl", "llm", "figlet", "file", "doc", "agent"]:
        versions.register_component(f"{forge_name}_forge", "0.4.0")
    diag.log_event("INFO", "Forge versions registered")

    # 4. Initialize Cognitive & Utility Forges with dependencies
    llm = LLMForge()
    memory = MemoryForge(persistence_path=FORGE_DIR / "memory_data.json", llm=llm)
    knowledge = KnowledgeForge(persistence_path=FORGE_DIR / "kb_data.json")
    code = CodeForge()
    refactor = RefactorForge()
    crawl = CrawlForge()
    figlet = FigletForge()
    file_f = FileForge(base_path=home_dir)
    doc = DocForge(base_path=home_dir)
    agent = AgentForge(llm=llm)
    
    diag.log_event("INFO", "All Forges initialized with persistent state")

    # 5. Initialize Type Forge and register core schemas
    types = TypeCore()
    
    eidosian_json_schema = {
        "type": "object",
        "properties": {
            "version": {"type": "string"},
            "principles": {"type": "object"},
            "metadata": {
                "type": "object",
                "properties": {
                    "creator": {"type": "string"},
                    "created": {"type": "string"}
                },
                "required": ["creator", "created"]
            }
        },
        "required": ["version", "principles", "metadata"]
    }
    types.register_schema("eidosian_json", eidosian_json_schema)
    diag.log_event("INFO", "Core schemas registered")

    # 4. Success
    gis.set("system.status", "active", persist=False)
    diag.log_event("INFO", "Foundation bootstrap complete")
    print("✅ Foundation active.")
    
    return gis, diag, types, versions, memory, knowledge, code, refactor, crawl, llm, figlet, file_f, doc, agent

if __name__ == "__main__":
    bootstrap()
