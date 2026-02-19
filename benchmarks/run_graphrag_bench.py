import os
import subprocess
import time
import shutil
import yaml
import urllib.request
import urllib.error
from pathlib import Path

# Configuration
MODELS_DIR = Path("models")
LLM_MODEL = MODELS_DIR / "qwen2.5-1.5b-instruct-q8_0.gguf"
LLM_MODEL_FALLBACK = MODELS_DIR / "Qwen2.5-0.5B-Instruct-Q8_0.gguf"
EMBED_MODEL = MODELS_DIR / "nomic-embed-text-v1.5.Q4_K_M.gguf"
LLAMA_SERVER_BIN = Path("llama.cpp/build/bin/llama-server")
INPUT_DATA_DIR = Path("data/graphrag_test/input")
WORKSPACE_DIR = Path("data/graphrag_test/workspace")
VENV_PYTHON = Path("eidosian_venv/bin/python3")
LOGS_DIR = Path("logs")

# Ports
LLM_PORT = 8081
EMBED_PORT = 8082


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
    cmd = [
        str(LLAMA_SERVER_BIN),
        "-m", str(model_path),
        "--port", str(port),
        "--ctx-size", "2048",
        "--n-gpu-layers", os.environ.get("EIDOS_LLAMA_GPU_LAYERS", "0"),
    ]
    if embedding:
        cmd.append("--embedding")
        cmd.append("--pooling")
        cmd.append("mean")
    else:
        # Optimization for chat
        cmd.append("--parallel")
        cmd.append("2")
        
    print(f"Starting server on port {port} with model {model_path}...")
    # Redirect output to file to keep console clean
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOGS_DIR / f"server_{port}.log", "w")
    process = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=_llama_env())

    # Wait for server to be ready.
    print(f"Waiting for server {port} to be ready...")
    wait_for_http(f"http://127.0.0.1:{port}/health", timeout_s=60.0)
    return process, log_file

def create_settings():
    settings = {
        "completion_models": {
            "default_completion_model": {
                "type": "litellm",
                "model_provider": "openai",
                "model": "qwen2.5-1.5b",
                "auth_method": "api_key",
                "api_key": "sk-no-key-required",
                "api_base": f"http://127.0.0.1:{LLM_PORT}/v1",
                "call_args": {
                    "temperature": 0.0,
                    "max_completion_tokens": 2048,
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

def run_benchmark():
    print("--- Starting GraphRAG Benchmark ---")
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
        
        # 2. Query Benchmark (Global)
        print("\n[Benchmark] Starting Global Query...")
        start_time = time.time()
        
        query_cmd = [
            str(VENV_PYTHON), "-m", "graphrag", "query",
            "--root", str(WORKSPACE_DIR),
            "--method", "global",
            "--query", "What is the relationship between Kael and Seraphina?"
        ]
        result = subprocess.run(query_cmd, capture_output=True, text=True)
        
        query_duration = time.time() - start_time
        print(f"[Benchmark] Global Query complete in {query_duration:.2f} seconds.")
        print(f"Output: {result.stdout.strip()}")
        
    except subprocess.CalledProcessError as e:
        print(f"Error during benchmark: {e}")
        # Only print stderr if available
        # subprocess.run raises CalledProcessError but stderr isn't captured unless capture_output=True
        # For index_cmd we didn't capture, so it's in stdout.
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        # Cleanup
        print("\nStopping servers...")
        llm_proc.terminate()
        embed_proc.terminate()
        llm_log.close()
        embed_log.close()

if __name__ == "__main__":
    run_benchmark()
