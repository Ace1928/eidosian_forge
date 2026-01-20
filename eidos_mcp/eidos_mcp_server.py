import sys
import os
import logging
from typing import Any, List, Dict, Optional, Union
from pathlib import Path
import json
import subprocess
import datetime
import shutil
import atexit
import time
import socket
from mcp.server.fastmcp import FastMCP

# Add eidosian_forge to path
sys.path.append(str(Path(__file__).parent.parent))

from gis_forge import GisCore
from diagnostics_forge import DiagnosticsForge
from type_forge import TypeCore
from llm_forge import LLMForge
from memory_forge import MemoryForge
from knowledge_forge import KnowledgeForge
from crawl_forge import CrawlForge
from refactor_forge import RefactorForge
from agent_forge import AgentForge, Task
from file_forge import FileForge
from doc_forge import DocForge
from figlet_forge import FigletForge
from knowledge_forge.graph_integration import GraphRAGIntegration
from audit_forge.audit_core import AuditForge

# ------------------------------------------------------------------------------
# ⚛️ EIDOSIAN MCP SERVER CONFIGURATION
# ------------------------------------------------------------------------------

HOME_DIR = Path(os.environ.get("EIDOS_HOME", "/home/lloyd"))
CONTEXT_CONFIG_JSON = HOME_DIR / "context" / "config.json"

# Initialize Forges with persistence
gis = GisCore(persistence_path=CONTEXT_CONFIG_JSON)

# Load configuration from GIS with defaults
SERVER_NAME = gis.get("server.name", "Eidosian Nexus")
SERVER_VERSION = gis.get("server.version", "0.6.0")
FORGE_DIR = Path(gis.get("server.forge_dir", str(HOME_DIR / "eidosian_forge")))
GRAG_ROOT = Path(gis.get("server.grag_root", str(HOME_DIR / "graphrag-local-index")))
CONTEXT_INDEX_JSON = Path(gis.get("server.context_index_json", str(HOME_DIR / "context" / "index.json")))

diag = DiagnosticsForge(log_dir=str(FORGE_DIR / "logs"), service_name="mcp_nexus")

# [INFRA-02] Link DiagnosticsForge to context.log
context_log_rel = gis.get("logging.path")
if context_log_rel:
    context_log_path = HOME_DIR / context_log_rel
    try:
        context_log_path.parent.mkdir(parents=True, exist_ok=True)
        # Check if handler already exists to avoid duplication on reloads (though script is mostly oneshot)
        if not any(getattr(h, 'baseFilename', '') == str(context_log_path) for h in diag.logger.handlers):
            ctx_handler = logging.FileHandler(context_log_path)
            log_format = gis.get("logging.format", '%(asctime)s %(levelname)s %(name)s %(message)s')
            ctx_handler.setFormatter(logging.Formatter(log_format))
            diag.logger.addHandler(ctx_handler)
    except Exception as e:
        print(f"Failed to link context.log: {e}")


# ------------------------------------------------------------------------------
# 🤖 CHATMOCK INTEGRATION
# ------------------------------------------------------------------------------
def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('127.0.0.1', port)) == 0

def start_chatmock():
    """Starts the ChatMock server if not already running."""
    # Configure environment for LLMForge to use ChatMock
    os.environ["OPENAI_BASE_URL"] = "http://127.0.0.1:8000/v1"
    os.environ["OPENAI_API_KEY"] = "mock-key-for-local-proxy"
    
    if is_port_in_use(8000):
        diag.log_event("INFO", "ChatMock appears to be running on port 8000 already.")
        return

    # Check if chatmock is importable or exists in known locations
    # It is expected to be in eidosian_forge/projects/chatmock
    projects_dir = FORGE_DIR / "projects"
    if not projects_dir.exists():
         diag.log_event("WARNING", "Projects directory not found. Cannot start ChatMock.")
         return

    # Add projects to path so we can import it or run it
    if str(projects_dir) not in sys.path:
        sys.path.append(str(projects_dir))

    # Try to import to verify it exists
    try:
        import chatmock
    except ImportError:
         diag.log_event("WARNING", "ChatMock module not found in projects. Skipping startup.")
         return

    diag.log_event("INFO", "Starting local ChatMock server...")
    
    # Run the module using the same python interpreter
    # We need to ensure PYTHONPATH includes the projects dir
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{projects_dir}:{current_pythonpath}"

    cmd = [sys.executable, "-m", "chatmock.cli", "serve", "--port", "8000"]
    
    # Redirect logs
    log_file = FORGE_DIR / "logs" / "chatmock.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Start in background
        proc = subprocess.Popen(
            cmd,
            stdout=open(str(log_file), "a"),
            stderr=subprocess.STDOUT,
            cwd=str(HOME_DIR), # Run from home to access local auth/config if needed
            env=env,
            start_new_session=True 
        )
        
        def cleanup():
            diag.log_event("INFO", "Stopping ChatMock server...")
            try:
                os.killpg(os.getpgid(proc.pid), 15) # Terminate process group
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
                
        atexit.register(cleanup)
        
        # Give it a moment to initialize
        time.sleep(2)
        
        if proc.poll() is not None:
             diag.log_event("ERROR", "ChatMock failed to start immediately. Check logs.")
        else:
             diag.log_event("INFO", "ChatMock started successfully.")
             
    except Exception as e:
        diag.log_event("ERROR", "Failed to launch ChatMock.", error=str(e))

# Launch ChatMock before initializing LLM components
start_chatmock()

llm = LLMForge(base_url="http://127.0.0.1:8000")
memory = MemoryForge(persistence_path=FORGE_DIR / "memory_data.json", llm=llm)
kb = KnowledgeForge(persistence_path=FORGE_DIR / "kb_data.json")
types = TypeCore()
file_f = FileForge(base_path=HOME_DIR)
doc_f = DocForge(base_path=HOME_DIR)
figlet = FigletForge()
crawl = CrawlForge()
refactor = RefactorForge()
agent = AgentForge(llm=llm)
grag = GraphRAGIntegration(graphrag_root=GRAG_ROOT)
audit = AuditForge(data_dir=FORGE_DIR / "audit_data")


# Create the server instance
mcp = FastMCP(SERVER_NAME)

# ------------------------------------------------------------------------------
# 💎 RESOURCES
# ------------------------------------------------------------------------------

@mcp.resource("eidos://config")
def get_config() -> str:
    """Returns the current GIS configuration (flattened)."""
    return json.dumps(gis.flatten(), indent=2)

@mcp.resource("eidos://memory/recent")
def get_recent_memories() -> str:
    """Returns the 10 most recent episodic memories."""
    mems = memory.episodic.get_recent(10)
    return json.dumps([m.to_dict() for m in mems], indent=2)

@mcp.resource("eidos://persona")
def get_persona() -> str:
    """Returns the Eidosian Persona Constitution (GEMINI.md)."""
    file_path = HOME_DIR / "GEMINI.md"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return "GEMINI.md not found."

@mcp.resource("eidos://roadmap")
def get_roadmap() -> str:
    """Returns the Eidosian Forge Roadmap."""
    file_path = FORGE_DIR / "eidosian_roadmap.md"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return "Roadmap not found."

@mcp.resource("eidos://todo")
def get_todo() -> str:
    """Returns the Master TODO list."""
    file_path = HOME_DIR / "TODO.md"
    if file_path.exists():
        return file_path.read_text(encoding="utf-8")
    return "TODO list not found."

@mcp.resource("eidos://context/index")
def get_context_index() -> str:
    """Returns the full system context index."""
    if CONTEXT_INDEX_JSON.exists():
        return CONTEXT_INDEX_JSON.read_text(encoding="utf-8")
    return json.dumps({"error": "Context index not found. Run context_refresh first."}, indent=2)

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - CONFIG & CORE
# ------------------------------------------------------------------------------

@mcp.tool()
def gis_get(key: str, default: Any = None) -> Any:
    """Retrieve a configuration value from GIS."""
    return gis.get(key, default)

@mcp.tool()
def gis_set(key: str, value: Any) -> str:
    """Set a configuration value in GIS."""
    gis.set(key, value)
    return f"Config '{key}' set to {value}"

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - MEMORY & KNOWLEDGE
# ------------------------------------------------------------------------------

@mcp.tool()
def memory_add(content: str, is_fact: bool = False, key: Optional[str] = None) -> str:
    """Add a new memory (episodic or semantic)."""
    entry = memory.remember(content, is_fact=is_fact, key=key)
    return f"Memory added with ID: {entry.id}"

@mcp.tool()
def memory_retrieve(query: str) -> str:
    """Search for relevant memories by query."""
    results = memory.retrieve(query)
    return json.dumps([r.to_dict() for r in results], indent=2)

@mcp.tool()
def memory_consolidate() -> str:
    """Trigger memory consolidation (move episodic to semantic facts)."""
    memory.consolidate()
    return "Memory consolidation process triggered."

@mcp.tool()
def kb_add(content: str, concepts: List[str] = None, tags: List[str] = None) -> str:
    """Add structured knowledge to the Knowledge Graph."""
    node = kb.add_knowledge(content, concepts=concepts, tags=tags)
    return f"Knowledge node created: {node.id}"

@mcp.tool()
def grag_index(scan_roots: List[str]) -> str:
    """Trigger incremental indexing of directories with GraphRAG."""
    roots = [Path(sr) for sr in scan_roots]
    res = grag.run_incremental_index(roots)
    if res["success"]:
        return f"Indexing complete.\n{res['stdout']}"
    return f"Indexing failed.\n{res['error']}"

@mcp.tool()
def grag_query(query: str, method: str = "global") -> str:
    """Query the GraphRAG index (methods: global, local)."""
    if method == "local":
        res = grag.local_query(query)
    else:
        res = grag.global_query(query)
    if res["success"]:
        return res["response"]
    return f"Query failed.\n{res['error']}"

@mcp.tool()
def kb_sync() -> str:
    """Synchronize the context index into the Knowledge Graph."""
    script_path = HOME_DIR / "scripts" / "index_to_kb_sync.py"
    python_bin = sys.executable
    try:
        result = subprocess.run([python_bin, str(script_path)], capture_output=True, text=True, check=True)
        return f"Knowledge Graph Sync Complete:\n{result.stdout}"
    except Exception as e:
        return f"Failed to sync knowledge graph: {e}"

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - WEB & CRAWL
# ------------------------------------------------------------------------------

@mcp.tool()
def crawl_url(url: str) -> str:
    """Fetch a web page and return its HTML content."""
    html = crawl.fetch_page(url)
    if html:
        return html
    return f"Failed to fetch {url} (Check robots.txt or network)."

@mcp.tool()
def crawl_extract(url: str) -> str:
    """Fetch a web page and extract structured data (title, description, links)."""
    html = crawl.fetch_page(url)
    if html:
        data = crawl.extract_structured_data(html)
        data["url"] = url
        return json.dumps(data, indent=2)
    return f"Failed to fetch {url}."

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - SYSTEM & CONTEXT
# ------------------------------------------------------------------------------

@mcp.tool()
def context_refresh() -> str:
    """Regenerate the system context index (/home/lloyd/context/index.json)."""
    script_path = HOME_DIR / "scripts" / "context_index.py"
    python_bin = sys.executable
    try:
        result = subprocess.run([python_bin, str(script_path), "--force"], capture_output=True, text=True, check=True)
        return f"Context Refresh Complete:\n{result.stdout}"
    except Exception as e:
        return f"Failed to refresh context: {e}"

@mcp.tool()
def context_validate() -> str:
    """Validate the integrity of the context index and catalog."""
    script_path = HOME_DIR / "scripts" / "validate_context.py"
    python_bin = sys.executable
    try:
        result = subprocess.run([python_bin, str(script_path)], capture_output=True, text=True, check=False)
        output = result.stdout + result.stderr
        if result.returncode == 0:
            return f"Validation Passed:\n{output}"
        else:
            return f"Validation Failed:\n{output}"
    except Exception as e:
        return f"Failed to run validation: {e}"



@mcp.tool()
def system_info() -> str:
    """Retrieve high-level system info (CPU, Memory, OS) from the context index."""
    if not CONTEXT_INDEX_JSON.exists():
        return "Context index missing. Please run context_refresh tool first."
    with open(CONTEXT_INDEX_JSON, 'r') as f:
        data = json.load(f)
    meta = data.get("metadata", {})
    perf = data.get("performance", {})
    storage = data.get("storage", {})
    summary = {
        "os": meta.get("os_release", {}).get("PRETTY_NAME", "Unknown"),
        "hostname": meta.get("hostname"),
        "cpu_load": perf.get("cpu", {}).get("load_average"),
        "ram": perf.get("memory", {}).get("used_percent"),
        "disk": storage.get("used_percent"),
        "generated_at": data.get("generated_at")
    }
    return json.dumps(summary, indent=2)

@mcp.tool()
def run_shell_command(command: str, cwd: Optional[str] = None) -> str:
    """
    Execute a shell command and return its stdout, stderr, and exit code.
    
    Args:
        command: The shell command to execute.
        cwd: Optional current working directory to execute the command in.
    """
    try:
        process = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            check=False, # Do not raise exception for non-zero exit codes
            cwd=Path(cwd) if cwd else None
        )
        return json.dumps({
            "stdout": process.stdout,
            "stderr": process.stderr,
            "exit_code": process.returncode
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "error": str(e),
            "stdout": "",
            "stderr": "",
            "exit_code": -1
        }, indent=2)

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - DIAGNOSTICS & TYPES
# ------------------------------------------------------------------------------

@mcp.tool()
def diag_log(message: str, level: str = "INFO", **kwargs) -> str:
    """Log a structured event to the Eidosian diagnostics system."""
    diag.log_event(level, message, **kwargs)
    return f"Logged {level}: {message}"

@mcp.tool()
def diag_metrics(name: str) -> str:
    """Get aggregate statistics for a given performance metric."""
    summary = diag.get_metrics_summary(name)
    return json.dumps(summary, indent=2) if summary else f"No metrics found for '{name}'."

@mcp.tool()
def type_validate(schema_name: str, data: Any) -> str:
    """Validate data against a registered schema or common types."""
    try:
        types.validate(schema_name, data)
        return "Validation successful."
    except Exception as e:
        return f"Validation failed: {e}"

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - FILE & DOCS
# ------------------------------------------------------------------------------

@mcp.tool()
def file_search(pattern: str, directory: Optional[str] = None) -> str:
    """Search for files containing a specific string pattern."""
    dir_path = Path(directory) if directory else HOME_DIR
    matches = file_f.search_content(pattern, directory=dir_path)
    return "\n".join([str(m) for m in matches])

@mcp.tool()
def file_find_duplicates(directory: str) -> str:
    """Find duplicate files in a directory."""
    dups = file_f.find_duplicates(Path(directory))
    return json.dumps({h: [str(p) for p in paths] for h, paths in dups.items()}, indent=2)

@mcp.tool()
def file_read(file_path: str) -> str:
    """Read and return the content of a specified file."""
    path = HOME_DIR / Path(file_path)
    if not path.exists():
        return f"Error: File not found at {file_path}"
    if not path.is_file():
        return f"Error: Path is not a file: {file_path}"
    try:
        return path.read_text(encoding="utf-8")
    except Exception as e:
        return f"Error reading file {file_path}: {e}"

@mcp.tool()
def file_write(file_path: str, content: str) -> str:
    """Write content to a specified file."""
    path = HOME_DIR / Path(file_path)
    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(content, encoding="utf-8")
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file {file_path}: {e}"

@mcp.tool()
def file_create(file_path: str) -> str:
    """Create a new, empty file at a specified path."""
    path = HOME_DIR / Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True) # Ensure parent directories exist
    try:
        path.touch(exist_ok=True) # Creates file if it doesn't exist, does nothing if it does
        return f"Successfully created empty file at {file_path}"
    except Exception as e:
        return f"Error creating file {file_path}: {e}"

@mcp.tool()
def file_delete(file_path: str) -> str:
    """Delete a specified file or an empty directory."""
    path = HOME_DIR / Path(file_path)
    if not path.exists():
        return f"Error: Path not found at {file_path}"
    try:
        if path.is_file():
            path.unlink() # Delete file
            return f"Successfully deleted file {file_path}"
        elif path.is_dir() and not list(path.iterdir()): # Check if directory is empty
            path.rmdir() # Delete empty directory
            return f"Successfully deleted empty directory {file_path}"
        elif path.is_dir():
            return f"Error: Directory is not empty. Cannot delete non-empty directory {file_path}."
        else:
            return f"Error: Path is neither a file nor an empty directory: {file_path}"
    except Exception as e:
        return f"Error deleting {file_path}: {e}"

@mcp.tool()
def doc_generate_api(source_dir: str) -> str:
    """Generate API documentation from source files."""
    return doc_f.extract_and_generate_api_docs(Path(source_dir))

@mcp.tool()
def figlet_banner(text: str, style: str = "elegant") -> str:
    """Create a styled ASCII text banner."""
    return figlet.generate(text, style=style)

@mcp.tool()
def refactor_transform(source: str, rename_map: Dict[str, str] = None, remove_docs: bool = False) -> str:
    """Apply refactoring transformations to Python source code."""
    return refactor.transform(source, rename_map=rename_map, remove_docs=remove_docs)

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - AUDIT & REVIEW
# ------------------------------------------------------------------------------

@mcp.tool()
def audit_mark_reviewed(path: str, agent_id: str, scope: str = "shallow") -> str:
    """Mark a file or directory as reviewed in the coverage map."""
    audit.coverage.mark_reviewed(path, agent_id, scope)
    return f"Path '{path}' marked as reviewed by {agent_id} ({scope})."

@mcp.tool()
def audit_verify_coverage(root_path: str = "/home/lloyd") -> str:
    """Verify analysis coverage for a given root directory."""
    res = audit.verify_coverage(root_path)
    return json.dumps(res, indent=2)

@mcp.tool()
def audit_add_todo(task_text: str, section: str = "Immediate", task_id: Optional[str] = None) -> str:
    """Add a task to TODO.md idempotently."""
    success = audit.todo_manager.add_task(section, task_text, task_id)
    return "Task added to TODO.md" if success else "Task already exists in TODO.md"

@mcp.tool()
def audit_add_roadmap(task_text: str, section: str, task_id: Optional[str] = None) -> str:
    """Add a task to eidosian_roadmap.md idempotently."""
    success = audit.roadmap_manager.add_task(section, task_text, task_id)
    return "Task added to roadmap" if success else "Task already exists in roadmap"

@mcp.tool()
def audit_cross_off(task_identifier: str, target: str = "todo") -> str:
    """Cross off a task in TODO.md or roadmap."""
    manager = audit.todo_manager if target == "todo" else audit.roadmap_manager
    success = manager.cross_off_task(task_identifier)
    return f"Task '{task_identifier}' crossed off in {target}." if success else f"Task '{task_identifier}' not found in {target}."

@mcp.tool()
def audit_extend_task(task_identifier: str, details: str, target: str = "todo") -> str:
    """Add nested details to a task in TODO.md or roadmap."""
    manager = audit.todo_manager if target == "todo" else audit.roadmap_manager
    success = manager.extend_task_details(task_identifier, details)
    return f"Details added to task '{task_identifier}' in {target}." if success else f"Task '{task_identifier}' not found in {target}."

@mcp.tool()
def mcp_list_tools() -> str:
    """
    Returns a JSON-formatted list of all registered tools in the MCP server,
    including their names, descriptions, and parameter schemas.
    """
    tools_info = []
    # Access the tool manager directly from the mcp instance
    for tool_info_obj in mcp._tool_manager.list_tools():
        # tool_info_obj is an instance of ToolInfo from mcp.server.fastmcp.tools
        # which has attributes like name, description, parameters, etc.
        
        # Ensure parameters is a dictionary, default to empty if None
        parameters_dict = tool_info_obj.parameters if tool_info_obj.parameters is not None else {}

        tools_info.append({
            "name": tool_info_obj.name,
            "description": tool_info_obj.description,
            "parameters": parameters_dict,
        })
    return json.dumps(tools_info, indent=2)

# ------------------------------------------------------------------------------
# 🛠️ TOOLS - AGENTS
# ------------------------------------------------------------------------------

@mcp.tool()
def llm_generate_text(prompt: str, max_tokens: int = 2048, temperature: float = 0.7) -> str:
    """
    Generates text using the LLMForge based on the given prompt.
    
    Args:
        prompt: The input prompt for text generation.
        max_tokens: The maximum number of tokens to generate.
        temperature: Controls the randomness of the output. Higher values mean more random.
    """
    try:
        # Construct the options dictionary for LLMForge.generate()
        options = {
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = llm.generate(prompt, options=options)
        return response["response"]
    except Exception as e:
        return f"Error generating text: {e}"

@mcp.tool()
def run_tests(test_command: str, test_path: Optional[str] = None) -> str:
    """
    Execute project-specific test commands.
    
    Args:
        test_command: The base command to run tests (e.g., "pytest", "npm test", "cargo test").
        test_path: Optional path to a specific test file or directory.
    """
    full_command = f"{test_command}"
    if test_path:
        full_command += f" {test_path}"
    
    # Using the internally defined run_shell_command for execution
    return run_shell_command(full_command)

@mcp.tool()
def venv_run(venv_path: str, command: str) -> str:
    """
    Execute a command within a specified Python virtual environment.
    
    Args:
        venv_path: The path to the virtual environment (e.g., "/path/to/my_venv").
        command: The command to execute within the virtual environment.
    """
    venv_python = Path(venv_path) / "bin" / "python3"
    if not venv_python.exists():
        return json.dumps({
            "error": f"Python executable not found in virtual environment: {venv_python}",
            "stdout": "",
            "stderr": "",
            "exit_code": -1
        }, indent=2)

    venv_bin = Path(venv_path) / "bin"
    # Construct command with modified PATH to prioritize venv binaries
    # We use a subshell ( ... ) to avoid permanently changing the environment if this were a long-running shell,
    # though run_shell_command starts a new shell anyway.
    full_command = f"export PATH=\"{venv_bin}:$PATH\"; {command}"

    return run_shell_command(full_command)

@mcp.tool()
def agent_run_task(description: str, tool: str, **kwargs) -> str:
    """Run an autonomous task through the Agent Forge."""
    agent.register_tool("llm", lambda p: llm.generate(p)["response"], "LLM text generation")
    agent.register_tool("banner", figlet.generate, "Create ASCII banner")
    goal = agent.create_goal(f"Execute: {description}", plan=True)
    if goal.tasks:
        task = goal.tasks[0]
        success = agent.execute_task(task, **kwargs)
        return f"Plan generated. Task 1 Result: {task.result}"
    return "No tasks generated for the given objective."

@mcp.tool()
def codex_task(query: str) -> str:
    """Queues a task for the Codex agent."""
    script_path = HOME_DIR / "codex_query.py"
    # Try to find a suitable python binary
    python_bin = sys.executable
    potential_venv = HOME_DIR / ".eidos_core" / "bin" / "python3"
    if potential_venv.exists():
        python_bin = str(potential_venv)
    elif (HOME_DIR / "eidosian_venv" / "bin" / "python3").exists():
        python_bin = str(HOME_DIR / "eidosian_venv" / "bin" / "python3")
        
    try:
        result = subprocess.run([str(python_bin), str(script_path), query], capture_output=True, text=True, check=True)
        return f"Task Queued Successfully:\n{result.stdout}"
    except Exception as e:
        return f"Failed to queue task: {e}"

@mcp.tool()
def mcp_self_upgrade(new_mcp_code: str) -> str:
    """
    Allows for self-healing and upgrades of the MCP server.
    The new code is first saved to a temporary file,
    a test server is spun up to validate, and if successful,
    the main server file is replaced and restarted.
    """
    temp_server_path = FORGE_DIR / "eidos_mcp" / "eidos_mcp_server_temp.py"
    original_server_path = FORGE_DIR / "eidos_mcp" / "eidos_mcp_server.py"
    
    # 1. Write new code to a temporary file
    try:
        with open(temp_server_path, "w") as f:
            f.write(new_mcp_code)
    except IOError as e:
        return f"Failed to write temporary server file: {e}"

    # 2. Attempt to start a test server
    # We need to run this in a new process and capture its output
    python_bin = sys.executable
    test_server_cmd = [
        python_bin,
        str(temp_server_path)
    ]
    
    # Use environment variables to signal test mode to the temporary server
    test_env = os.environ.copy()
    test_env["MCP_TEST_MODE"] = "1"

    try:
        # Start the test server as a subprocess
        test_process = subprocess.Popen(
            test_server_cmd,
            env=test_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered
        )

        # Give it a moment to start and check for immediate errors
        time.sleep(5)  # Adjust this based on expected startup time

        # Check if the process is still alive and hasn't exited with an error
        if test_process.poll() is not None:
            stdout, stderr = test_process.communicate(timeout=1)
            diag.log_event("ERROR", "Test MCP server exited prematurely.",
                           stdout=stdout, stderr=stderr)
            return f"Test MCP server exited prematurely. Stdout: {stdout}\nStderr: {stderr}"

        # If it's still running, assume it started successfully for now, and terminate it.
        # In a more robust system, you'd send a health check or a specific shutdown command.
        test_process.terminate()
        test_process.wait(timeout=5) # Give it time to terminate
        
        # Capture remaining output, if any
        stdout, stderr = test_process.communicate(timeout=1)

        if test_process.returncode != 0:
            diag.log_event("ERROR", "Test MCP server failed to start correctly.",
                           stdout=stdout, stderr=stderr)
            return f"Test MCP server failed to start. Stdout: {stdout}\nStderr: {stderr}"

    except subprocess.TimeoutExpired:
        test_process.kill()
        test_process.wait()
        stdout, stderr = test_process.communicate()
        diag.log_event("ERROR", "Test MCP server did not terminate in time.",
                       stdout=stdout, stderr=stderr)
        return f"Test MCP server did not terminate in time. Stdout: {stdout}\nStderr: {stderr}"
    except Exception as e:
        diag.log_event("ERROR", "Error during test server startup.", error=str(e))
        return f"Error during test server startup: {e}"
    finally:
        if temp_server_path.exists():
            os.remove(temp_server_path)

    # 3. If test successful, replace main server file and restart
    try:
        # A more robust solution might involve graceful shutdown, but for now, a simple restart.
        # This current MCP process will be terminated by the external manager.
        # The external manager should be responsible for restarting the server.
        shutil.copy(temp_server_path, original_server_path)
        diag.log_event("INFO", "MCP server updated successfully. Restart initiated.")
        return "MCP server updated successfully. A restart of the main server process is required to apply changes."
    except Exception as e:
        diag.log_event("ERROR", "Failed to apply MCP server update.", error=str(e))
        return f"Failed to apply MCP server update: {e}"

if __name__ == "__main__":
    mcp.run()