import os
import subprocess
import time
import shutil
import yaml
import json
import argparse
import cProfile
import pstats
import io
import pandas as pd
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone

# Configuration
MODELS_DIR = Path("models")
MODEL_SELECTION_PATH = Path(os.environ.get("EIDOS_MODEL_SELECTION_PATH", "config/model_selection.json"))


def _load_model_selection() -> dict:
    if not MODEL_SELECTION_PATH.exists():
        return {}
    try:
        payload = json.loads(MODEL_SELECTION_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _selection_model(path: str, fallback: str) -> str:
    payload = _load_model_selection()
    services = payload.get("services") if isinstance(payload, dict) else None
    graphrag = services.get("graphrag") if isinstance(services, dict) else None
    if isinstance(graphrag, dict):
        value = str(graphrag.get(path, "")).strip()
        if value:
            return value
    return fallback


LLM_MODEL = Path(
    os.environ.get(
        "EIDOS_GRAPHRAG_LLM_MODEL",
        _selection_model("completion_model", str(MODELS_DIR / "Qwen2.5-0.5B-Instruct-Q8_0.gguf")),
    )
)
LLM_MODEL_FALLBACK = Path(
    os.environ.get(
        "EIDOS_GRAPHRAG_LLM_MODEL_FALLBACK",
        _selection_model("completion_fallback", str(MODELS_DIR / "Llama-3.2-1B-Instruct-Q8_0.gguf")),
    )
)
EMBED_MODEL = Path(
    os.environ.get(
        "EIDOS_GRAPHRAG_EMBED_MODEL",
        _selection_model("embedding_model", str(MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf")),
    )
)
LLAMA_SERVER_BIN = Path("llama.cpp/build/bin/llama-server")
INPUT_DATA_DIR = Path(os.environ.get("EIDOS_GRAPHRAG_INPUT_DIR", "data/graphrag_test/input"))
WORKSPACE_DIR = Path(os.environ.get("EIDOS_GRAPHRAG_WORKSPACE_DIR", "data/graphrag_test/workspace"))
VENV_PYTHON = Path("eidosian_venv/bin/python3")
LOGS_DIR = Path("logs")
REPORTS_DIR = Path(os.environ.get("EIDOS_GRAPHRAG_REPORTS_DIR", "reports/graphrag"))

# Ports
def _registry_port(service_key: str, fallback: int) -> int:
    registry_path = Path("config/ports.json")
    if not registry_path.exists():
        return fallback
    try:
        payload = json.loads(registry_path.read_text())
    except Exception:
        return fallback
    services = payload.get("services") if isinstance(payload, dict) else None
    if not isinstance(services, dict):
        return fallback
    service = services.get(service_key)
    if not isinstance(service, dict):
        return fallback
    try:
        value = int(service.get("port", fallback))
    except Exception:
        return fallback
    return value if value > 0 else fallback


LLM_PORT = int(os.environ.get("EIDOS_GRAPHRAG_LLM_PORT", str(_registry_port("graphrag_llm", 8081))))
EMBED_PORT = int(os.environ.get("EIDOS_GRAPHRAG_EMBED_PORT", str(_registry_port("graphrag_embedding", 8082))))
QUERY_METHOD = os.environ.get("EIDOS_GRAPHRAG_QUERY_METHOD", "global")

PLACEHOLDER_MARKERS = ("auto-generated placeholder", "fallback generated")
EXPECTED_STORY_TERMS = ("alaric", "seraphina", "kael", "malakar", "eidos", "crystal", "whispering caves")
NON_ANSWER_MARKERS = (
    "i am sorry",
    "unable to answer",
    "insufficient",
    "i don't know",
    "i do not know",
)
COMMUNITY_REPORT_TEXT_PROMPT_MINIMAL = """
You are generating a strict JSON community report from provided source text.

Rules:
- Use only facts from the provided Text.
- Never use Enron/example/template content.
- If information is absent, state that clearly instead of inventing.
- Keep output concise and grounded.
- Return exactly one JSON object and no prose outside JSON.
- Use short plain strings and avoid nested quotes inside values.

Return a single valid JSON object with keys:
- title (string)
- summary (string)
- rating (number 0-10)
- rating_explanation (string)
- findings (array of 3 to 5 objects with keys summary and explanation)
- Findings should cover key actors, conflict, and relationship details present in Text.
- Include named entities from Text when evidence exists.
- Use entity names exactly as they appear in Text.

Text:
{input_text}

Output JSON:
"""

COMMUNITY_REPORT_GRAPH_PROMPT_MINIMAL = """
You are generating a strict JSON community report from entity/relationship context.

Rules:
- Use only facts from the provided Text block.
- Never use tutorial/example/template content.
- Mention concrete entities from the Text in title, summary, and findings.
- If information is missing, say "insufficient evidence" for that point.
- Return exactly one JSON object and no prose outside JSON.
- Use short plain strings and avoid nested quotes inside values.
- Use 3 to 5 findings and preserve relationship details.

Return one valid JSON object:
{
  "title": "...",
  "summary": "...",
  "rating": <0-10>,
  "rating_explanation": "...",
  "findings": [{"summary":"...","explanation":"..."}]
}

Text:
{input_text}

Output JSON:
"""


def resolve_model(primary: Path, fallback: Path | None = None) -> Path:
    if primary.exists():
        return primary
    if fallback and fallback.exists():
        return fallback
    raise FileNotFoundError(f"Missing model: {primary}" + (f" (fallback: {fallback})" if fallback else ""))


def _llama_env() -> dict:
    env = os.environ.copy()
    bin_dir = str(LLAMA_SERVER_BIN.resolve().parent)
    env["PATH"] = f"{bin_dir}:{env.get('PATH', '')}"
    ld_library = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = f"{bin_dir}:{ld_library}" if ld_library else bin_dir
    return env


def wait_for_http(url: str, timeout_s: float = 30.0) -> None:
    start = time.time()
    while time.time() - start < timeout_s:
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return
        except Exception:
            time.sleep(0.5)
    raise TimeoutError(f"Timed out waiting for {url}")

def setup_data():
    if INPUT_DATA_DIR.exists():
        shutil.rmtree(INPUT_DATA_DIR)
    INPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create sample data - A small story for graph extraction
    story = """
    The Kingdom of Eidos was ruled by King Alaric. 
    King Alaric had a trusted advisor named Seraphina.
    Seraphina was a master of the Arcane Arts, specifically the discipline of Chronomancy.
    The enemy of Eidos was the Shadow Dominion, led by the dark sorcerer Malakar.
    Malakar sought the Crystal of Eternity, which was hidden in the Whispering Caves.
    Alaric sent his brave knight, Sir Kael, to protect the Crystal.
    Kael was secretly in love with Seraphina.
    """
    (INPUT_DATA_DIR / "story.txt").write_text(story)
    
    if WORKSPACE_DIR.exists():
        shutil.rmtree(WORKSPACE_DIR)
    (WORKSPACE_DIR / "input").mkdir(parents=True, exist_ok=True)
    (WORKSPACE_DIR / "input" / "story.txt").write_text(story)

def start_server(model_path, port, embedding=False):
    ctx_size = os.environ.get("EIDOS_LLAMA_CTX_SIZE", "4096")
    temperature = os.environ.get("EIDOS_LLAMA_TEMPERATURE", "0")
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
        "--ctx-size", str(ctx_size),
        "--temp", str(temperature),
        "--n-gpu-layers", os.environ.get("EIDOS_LLAMA_GPU_LAYERS", "0"),
    ]
    if embedding:
        cmd.append("--embedding")
        cmd.append("--pooling")
        cmd.append("mean")
    else:
        # Keep parallel at 1 so each slot gets full ctx-size for GraphRAG prompts.
        parallel_slots = os.environ.get("EIDOS_LLAMA_PARALLEL", "1")
        cmd.append("--parallel")
        cmd.append(str(parallel_slots))
        
    print(f"Starting server on port {port} with model {model_path}...")
    # Redirect output to file to keep console clean
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOGS_DIR / f"server_{port}.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=_llama_env())

    # Wait for server to be ready.
    print(f"Waiting for server {port} to be ready...")
    wait_for_http(f"http://127.0.0.1:{port}/health", timeout_s=60.0)
    return process, log_file


def validate_pipeline_outputs() -> dict:
    output_dir = WORKSPACE_DIR / "output"
    reports_path = output_dir / "community_reports.parquet"
    if not reports_path.exists():
        raise RuntimeError(f"Missing community reports output: {reports_path}")
    reports_df = pd.read_parquet(reports_path)
    if reports_df.empty:
        raise RuntimeError("Community reports output is empty.")

    placeholder_rows = []
    relevance_fail_rows = []
    schema_fail_rows = []
    for idx, row in reports_df.iterrows():
        summary = str(row.get("summary", "")).lower()
        full_content = str(row.get("full_content", "")).lower()
        if any(marker in summary or marker in full_content for marker in PLACEHOLDER_MARKERS):
            placeholder_rows.append(idx)
        term_hits = [term for term in EXPECTED_STORY_TERMS if term in summary or term in full_content]
        if len(term_hits) < 2:
            relevance_fail_rows.append({"row": int(idx), "term_hits": term_hits})
        findings = row.get("findings", None)
        title = str(row.get("title", "")).strip()
        rating = row.get("rating", row.get("rank", None))
        if not title or findings is None or pd.isna(rating):
            schema_fail_rows.append(
                {
                    "row": int(idx),
                    "title_present": bool(title),
                    "has_findings": findings is not None,
                    "has_rating": not pd.isna(rating) if rating is not None else False,
                }
            )

    if placeholder_rows:
        raise RuntimeError(
            f"Placeholder/fallback community reports detected at rows {placeholder_rows}; "
            "GraphRAG output is not production-ready."
        )
    if relevance_fail_rows:
        raise RuntimeError(
            "Community report relevance validation failed. Expected story entities missing: "
            f"{relevance_fail_rows}"
        )
    if schema_fail_rows:
        raise RuntimeError(
            "Community report schema validation failed for rows: "
            f"{schema_fail_rows}"
        )

    return {
        "community_reports_rows": int(len(reports_df)),
        "placeholder_rows": int(len(placeholder_rows)),
        "relevance_fail_rows": int(len(relevance_fail_rows)),
        "schema_fail_rows": int(len(schema_fail_rows)),
    }

def create_settings():
    prompts_dir = WORKSPACE_DIR / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    text_prompt_path = (prompts_dir / "community_report_text_minimal.txt").resolve()
    graph_prompt_path = (prompts_dir / "community_report_graph_minimal.txt").resolve()
    text_prompt_path.write_text(COMMUNITY_REPORT_TEXT_PROMPT_MINIMAL.strip() + "\n")
    graph_prompt_path.write_text(COMMUNITY_REPORT_GRAPH_PROMPT_MINIMAL.strip() + "\n")

    settings = {
        "completion_models": {
            "default_completion_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": "qwen2.5-0.5b-local",
                "auth_method": "api_key",
                "api_key": "sk-no-key-required",
                "api_base": f"http://127.0.0.1:{LLM_PORT}/v1",
                "call_args": {
                    "temperature": 0.0,
                    "max_completion_tokens": int(os.environ.get("EIDOS_GRAPHRAG_MAX_COMPLETION_TOKENS", "512")),
                    "max_tokens": int(os.environ.get("EIDOS_GRAPHRAG_MAX_TOKENS", "512")),
                },
            }
        },
        "embedding_models": {
            "default_embedding_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": "nomic-embed-text",
                "auth_method": "api_key",
                "api_key": "sk-no-key-required",
                "api_base": f"http://127.0.0.1:{EMBED_PORT}/v1",
                "call_args": {
                    "user": "graphrag",
                    "encoding_format": "float",
                },
            }
        },
        "concurrent_requests": 1,
        "async_mode": "asyncio",
        "chunking": {
            "type": "tokens",
            "size": 300,
            "overlap": 100,
            "encoding_model": "o200k_base",
        },
        "input": {
            "type": "text",
            "encoding": "utf-8",
            "file_pattern": ".*\\.txt$$"
        },
        "input_storage": {
            "type": "file",
            "base_dir": "input",
        },
        "output_storage": {
            "type": "file",
            "base_dir": "output",
        },
        "update_output_storage": {
            "type": "file",
            "base_dir": "update_output",
        },
        "cache": {
            "type": "json",
            "storage": {
                "type": "file",
                "base_dir": "cache",
            }
        },
        "reporting": {
            "type": "file",
            "base_dir": "logs"
        },
        "embed_text": {
            "embedding_model_id": "default_embedding_model",
        },
        "extract_graph": {
            "completion_model_id": "default_completion_model",
        },
        "summarize_descriptions": {
            "completion_model_id": "default_completion_model",
        },
        "community_reports": {
            "completion_model_id": "default_completion_model",
            "graph_prompt": str(graph_prompt_path),
            "text_prompt": str(text_prompt_path),
            "max_length": 400,
            "max_input_length": 4000,
        },
        "prune_graph": {
            "min_node_freq": 1,
            "min_node_degree": 0,
            "min_edge_weight_pct": 0.0,
            "remove_ego_nodes": False,
            "lcc_only": False,
        },
        "local_search": {
            "completion_model_id": "default_completion_model",
            "embedding_model_id": "default_embedding_model",
        },
        "global_search": {
            "completion_model_id": "default_completion_model",
        },
        "workflows": [
            "load_input_documents",
            "create_base_text_units",
            "create_final_documents",
            "extract_graph_nlp",
            "prune_graph",
            "finalize_graph",
            "create_communities",
            "create_final_text_units",
            "create_community_reports_text",
        ],
    }
    
    with open(WORKSPACE_DIR / "settings.yaml", "w") as f:
        yaml.dump(settings, f)

def run_benchmark(query: str) -> dict:
    print("--- Starting GraphRAG Benchmark ---")
    if os.environ.get("EIDOS_GRAPHRAG_ALLOW_PLACEHOLDER", "0").strip().lower() in {"1", "true", "yes", "on"}:
        raise RuntimeError("EIDOS_GRAPHRAG_ALLOW_PLACEHOLDER must be disabled for production benchmark runs.")
    llm_model = resolve_model(LLM_MODEL, LLM_MODEL_FALLBACK)
    embed_model = resolve_model(EMBED_MODEL)
    setup_data()
    create_settings()
    
    # Start Servers
    llm_proc, llm_log = start_server(llm_model, LLM_PORT, embedding=False)
    embed_proc, embed_log = start_server(embed_model, EMBED_PORT, embedding=True)
    
    try:
        # 1. Indexing Benchmark
        print("\n[Benchmark] Starting Indexing...")
        start_time = time.time()
        
        # Using module execution instead of direct cli bin which might not be in path
        index_cmd = [
            str(VENV_PYTHON),
            "-m",
            "graphrag",
            "index",
            "--root",
            str(WORKSPACE_DIR),
            "--method",
            "fast",
        ]
        subprocess.run(index_cmd, check=True)
        
        index_duration = time.time() - start_time
        print(f"[Benchmark] Indexing complete in {index_duration:.2f} seconds.")
        output_quality = validate_pipeline_outputs()
        print(
            f"[Benchmark] Output validation complete: "
            f"community_reports_rows={output_quality['community_reports_rows']}, "
            f"placeholder_rows={output_quality['placeholder_rows']}, "
            f"relevance_fail_rows={output_quality['relevance_fail_rows']}."
        )
        
        # 2. Query Benchmark
        print(f"\n[Benchmark] Starting {QUERY_METHOD.title()} Query...")
        start_time = time.time()
        
        query_cmd = [
            str(VENV_PYTHON),
            "-m",
            "graphrag",
            "query",
            "--root",
            str(WORKSPACE_DIR),
            "--method",
            QUERY_METHOD,
            "--response-type",
            "Single Sentence",
            query,
        ]
        result = subprocess.run(query_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"GraphRAG query failed with exit {result.returncode}: {result.stderr.strip() or result.stdout.strip()}"
            )

        query_duration = time.time() - start_time
        raw_query_text = result.stdout.strip() or result.stderr.strip()
        filtered_lines = []
        for line in raw_query_text.splitlines():
            trimmed = line.strip()
            if not trimmed:
                continue
            if trimmed.startswith("DEBUG:"):
                continue
            if trimmed.startswith("💎 "):
                continue
            filtered_lines.append(trimmed)
        query_text = "\n".join(filtered_lines) if filtered_lines else raw_query_text
        lowered_query = query_text.lower()
        if any(marker in lowered_query for marker in NON_ANSWER_MARKERS):
            raise RuntimeError(f"Query output is non-informative: {query_text}")
        print(f"[Benchmark] {QUERY_METHOD.title()} Query complete in {query_duration:.2f} seconds.")
        print(f"Output: {query_text}")
        benchmark_result = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "llm_model": str(llm_model),
            "embed_model": str(embed_model),
            "index_seconds": round(index_duration, 3),
            "query_seconds": round(query_duration, 3),
            "query": query,
            "query_method": QUERY_METHOD,
            "query_output": query_text,
            "query_output_chars": len(query_text),
            "workspace_root": str(WORKSPACE_DIR),
            "output_quality": output_quality,
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = REPORTS_DIR / f"bench_metrics_{stamp}.json"
        report_path.write_text(json.dumps(benchmark_result, indent=2) + "\n")
        print(f"[Benchmark] Metrics saved to {report_path}")
        return benchmark_result
        
    except subprocess.CalledProcessError as e:
        print(f"Error during benchmark: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise
    finally:
        # Cleanup
        print("\nStopping servers...")
        llm_proc.terminate()
        embed_proc.terminate()
        try:
            llm_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            llm_proc.kill()
        try:
            embed_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            embed_proc.kill()
        llm_log.close()
        embed_log.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GraphRAG benchmark with local llama.cpp backends.")
    parser.add_argument(
        "--query",
        default="What is the relationship between Kael and Seraphina?",
        help="Query prompt for the benchmark.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable cProfile and save profile + top functions report.",
    )
    args = parser.parse_args()

    if args.profile:
        profiler = cProfile.Profile()
        profiler.enable()
        run_benchmark(args.query)
        profiler.disable()
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prof_path = REPORTS_DIR / f"bench_profile_{stamp}.prof"
        txt_path = REPORTS_DIR / f"bench_profile_{stamp}.txt"
        profiler.dump_stats(prof_path)
        stats_buffer = io.StringIO()
        pstats.Stats(profiler, stream=stats_buffer).sort_stats("cumulative").print_stats(40)
        txt_path.write_text(stats_buffer.getvalue())
        print(f"[Benchmark] Profile saved to {prof_path}")
        print(f"[Benchmark] Profile summary saved to {txt_path}")
    else:
        run_benchmark(args.query)
