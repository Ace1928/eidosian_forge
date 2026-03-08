#!/usr/bin/env python3
import os
from pathlib import Path

def replace_in_file(path_str):
    p = Path(path_str)
    if not p.exists():
        return
    text = p.read_text(encoding='utf-8')
    orig = text
    
    text = text.replace("Qwen2.5-1.5B-Instruct", "Qwen3.5-2B-Instruct")
    text = text.replace("qwen2.5:1.5b", "qwen3.5:2b")
    text = text.replace("Qwen 2.5 1.5B", "Qwen 3.5 2B")
    text = text.replace("qwen-1.5b", "qwen-3.5-2b")
    text = text.replace("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen3.5-2B-Instruct")
    text = text.replace("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen3.5-2B-Instruct")
    text = text.replace("Qwen2.5-1.5B-Instruct-Q8_0.gguf", "Qwen3.5-2B-Instruct-Q4_K_M.gguf")
    
    if orig != text:
        p.write_text(text, encoding='utf-8')
        print(f"Updated {p}")

targets = [
    "README.md",
    "FORGE_STATUS.md",
    "ARCHITECTURE.md",
    "docs/MODEL_SELECTION.md",
    "viz_forge/README.md",
    "scripts/living_knowledge_pipeline.py",
    "scripts/context_catalog.py",
    "word_forge/scripts/verify_llm.py",
    "word_forge/src/word_forge/parser/parser_refiner.py",
    "word_forge/living_lexicon_discussion.md",
    "benchmarks/run_graphrag_bench.py",
    "benchmarks/eidosian_consensus_bench.py",
    "benchmarks/graphrag_qualitative_assessor.py",
    "benchmarks/README.md",
    "benchmarks/model_domain_suite.py",
    "benchmarks/tests/test_graphrag_model_sweep.py",
    "agent_forge/notebooks/smolagents_examples.py",
    "agent_forge/notebooks/working_document.py",
    "eidos_mcp/src/eidos_mcp/config/models.py",
    "eidos_mcp/tests/test_model_config_runtime_defaults.py",
    "eidos_mcp/tests/test_mcp_tool_calls_individual.py"
]

for t in targets:
    replace_in_file(f"/data/data/com.termux/files/home/eidosian_forge/{t}")

print("Update complete.")
