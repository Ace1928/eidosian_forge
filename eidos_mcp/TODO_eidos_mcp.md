# Eidosian MCP Enhancement TODO List

This document outlines a series of tasks to extend, enhance, and significantly improve the `eidos_mcp` server and its related components. The goal is to improve robustness, testability, maintainability, performance, and overall developer experience.

## 1. Core Server (`eidos_mcp_server.py`) Improvements

### 1.1. Architecture & Design
- [ ] **Refactor Forge Initialization:** Introduce a dedicated `ForgeManager` class or function to centralize forge initialization. This should enable easier dependency injection, configuration management, and potentially lazy loading of forges.
- [x] **Configuration Management:** Implement a proper configuration system (e.g., `pydantic` settings, a dedicated `config.py` module, or loading from a YAML/TOML file) for server-wide parameters like `SERVER_NAME`, `SERVER_VERSION`, `HOME_DIR`, `FORGE_DIR`, `GRAG_ROOT`, and `CONTEXT_INDEX_JSON`, instead of hardcoded globals.
- [ ] **Forge Abstraction:** Define interfaces or abstract base classes for Forges (e.g., `BaseForge`) to ensure consistency, facilitate type checking, and allow for easier swapping of implementations or mocking in tests.

### 1.2. Error Handling & Robustness
- [ ] **Standardized Error Responses:** Implement a consistent, structured error response format (e.g., JSON objects with `code`, `message`, `details` fields) across all tools. This might involve custom exceptions that `FastMCP` can catch and serialize.
- [ ] **Subprocess Management Enhancement (`mcp_self_upgrade`, `context_refresh`, `codex_task`):**
    - [ ] Improve `subprocess.run` and `subprocess.Popen` calls with more robust error detection, including checking return codes, capturing `stderr`, and implementing stricter timeouts for `communicate()`.
    - [ ] Add explicit checks for script existence (e.g., `script_path.exists()`) before attempting to run external scripts.
- [x] ~~**Resource Not Found Handling:** `eidos://persona`, `eidos://roadmap`, `eidos://todo`, and `eidos://context/index` now return informative JSON error objects when source files are missing.~~
- [ ] **Input Validation for Tools:** Implement thorough input validation for all tool arguments, leveraging `type_forge` where appropriate, to prevent unexpected behavior, errors, or potential security vulnerabilities.

### 1.3. Performance & Asynchronicity
- [ ] **Identify Blocking Operations:** Profile or carefully review all Forge methods and tool implementations to identify potentially long-running or blocking I/O-bound operations (e.g., network requests, complex file operations, extensive LLM calls, GraphRAG processing).
- [ ] **Asynchronous Tool Execution:** If `FastMCP` supports it, convert identified blocking tools to use `async/await` patterns or offload them to a separate worker process/thread pool to prevent blocking the main server loop.

### 1.4. Logging & Observability
- [ ] **Comprehensive Internal Server Logging:** Add more detailed logging using the `diag` forge for key internal operations within the MCP server, such as:
    - Server startup and shutdown events.
    - Tool and resource registration.
    - Incoming requests and their corresponding tool/resource calls.
    - Internal errors or warnings.
- [ ] **Traceability:** Consider adding request IDs or correlation IDs to log messages to trace a single request's lifecycle through various components.

### 1.5. Security & Safety
- [ ] **Authentication/Authorization for Sensitive Tools:** Implement robust authentication and authorization checks for critical tools, especially `mcp_self_upgrade`, to prevent unauthorized code execution or system modifications.
- [ ] **Atomic File Operations:** For `mcp_self_upgrade`, ensure the file replacement using `shutil.copy` is atomic. Consider using `os.replace` (Python 3.3+) for safer file swaps to prevent corruption during unexpected interruptions.
- [ ] **Version Rollback Mechanism:** Develop a mechanism for `mcp_self_upgrade` to automatically or manually rollback to a previous working version if a new version fails validation or causes issues after deployment.

### 1.6. Tool & Resource Definition
- [ ] **Enhanced Tool Metadata:** Explore adding more explicit metadata to tool definitions beyond just docstrings (e.g., `pydantic` models for input/output schemas, examples, detailed descriptions). This would benefit automated clients and future UI generation.
- [ ] **Automatic OpenAPI/Swagger Generation:** Investigate if `FastMCP` can automatically generate OpenAPI/Swagger documentation from the registered tools and resources to provide a self-documenting API.

### 1.7. Agent Integration
- [ ] **Standardized Agent Tool Registration:** Refactor `agent_run_task` to dynamically discover and register *all* available MCP tools with `AgentForge`, rather than hardcoding a select few. This could involve iterating through `mcp.tool_registry`.

## 2. Test Suite (`test_nexus.py`) Enhancements

### 2.1. Test Coverage Expansion
- [ ] **Comprehensive Tool Test Coverage:** Add integration tests for *every single tool* exposed by `eidos_mcp_server.py`. Each tool should have at least one positive test case and relevant negative test cases.
    - [ ] `gis_get`, `gis_set`
    - [ ] `memory_retrieve`, `memory_consolidate`
    - [ ] `kb_add`
    - [ ] `grag_index`, `grag_query`
    - [ ] `crawl_url`, `crawl_extract`
    - [ ] `context_refresh`, `system_info`
    - [ ] `diag_log`, `diag_metrics`
    - [ ] `type_validate`
    - [ ] `file_search`, `file_find_duplicates`
    - [ ] `doc_generate_api`
    - [ ] `figlet_banner`
    - [ ] `refactor_transform`
    - [ ] `audit_mark_reviewed`, `audit_verify_coverage`, `audit_add_todo`, `audit_add_roadmap`, `audit_cross_off`, `audit_extend_task`
    - [ ] `agent_run_task`, `codex_task`
    - [ ] `mcp_self_upgrade` (requires careful mocking/isolation)
- [ ] **Resource Content Verification:** For `eidos://config` and `eidos://context/index`, verify the structure and presence of expected data, not just that content is returned.
- [ ] **Negative Test Cases:** Add tests to verify that tools and resources handle invalid inputs, non-existent data, and error conditions gracefully, returning expected error messages or codes.

### 2.2. Test Environment Isolation & Setup
- [x] ~~**Dynamic Server Script Path:** `test_nexus.py` derives interpreter and module startup dynamically (`PYTHON_BIN` + `-m eidos_mcp.eidos_mcp_server`) with environment overrides.~~
- [ ] **Isolated Test Environment for Persistence:**
    - [ ] Modify `eidos_mcp_server.py` to allow Forges to accept optional temporary paths for their persistence files (e.g., `persistence_path=FORGE_DIR / "gis_data.json"`) during testing.
    - [ ] In `TestEidosianNexus.asyncSetUp`, create a unique temporary directory for each test run.
    - [ ] Pass these temporary paths to the `StdioServerParameters.env` dictionary to be consumed by the server under test.
- [ ] **Proper Cleanup:** Implement robust `async tearDown` methods to ensure that all temporary files, directories, or server processes created during tests are properly cleaned up, preventing test interference and resource leaks.
- [x] ~~**Replace `run_server.sh` in Tests:** Tests now directly invoke `python -m eidos_mcp.eidos_mcp_server` via `StdioServerParameters` with explicit test transport env.~~

### 2.3. Test Reliability & Maintainability
- [x] ~~**Update Tool Name:** Changed tests to use `session.call_tool("memory_add", ...)` in `test_tool_remember`, matching the current MCP tool name.~~
- [x] ~~**Configurable Latency Thresholds:** Startup latency threshold now uses `EIDOS_MCP_TEST_STARTUP_MAX_SEC` (default `35.0`) in `test_nexus.py`.~~

## 3. Resource Fetcher (`eidos_fetch.py`) Enhancements

### 3.1. Portability & Configuration
- [x] ~~**Dynamic Server Script Path:** `eidos_fetch.py` now invokes the server via module entry (`python -m eidos_mcp.eidos_mcp_server`) with configurable interpreter (`--python` / `EIDOS_PYTHON_BIN`), avoiding hardcoded script paths.~~
- [x] ~~**Dynamic `PYTHONPATH`:** `eidos_fetch.py` dynamically assembles `PYTHONPATH` from forge root, `eidos_mcp/src`, and `lib` for portable stdio startup.~~

### 3.2. Usability & Output
- [x] ~~**Structured Error Output:** Added structured error payload emission via `--json-errors` (and `--json`) for programmatic consumption.~~
- [x] ~~**Handle Diverse Resource Content:** Fetcher now normalizes text/blob resource content (including binary-safe blob encoding metadata) and emits non-text resource summaries safely.~~
- [x] ~~**Add `--list` Option:** Added `--list` mode to enumerate available MCP resources over stdio/http transports.~~
- [x] ~~**Add `--json` Output Option:** Added `--json` output mode for both resource listing and resource reads.~~

### 3.3. Dependency Management
- [x] ~~**Direct Python Invocation:** `eidos_fetch.py` directly launches MCP via Python module invocation and no longer depends on `run_server.sh`.~~

## 4. Diagnostics (`check_mcp.py`) Improvements

### 4.1. Expanded Diagnostic Checks
- [x] ~~**Comprehensive Component Check:** `check_mcp.py` now verifies MCP package version, critical MCP imports, and expected environment variable presence, with strict-mode failure gating.~~
    - Verifying the installed `mcp` package version.
    - Attempting to import `ClientSession`, `StdioServerParameters`, and other critical classes.
    - Checking for expected environment variables (e.g., `EIDOS_HOME_DIR`, `EIDOS_FORGE_DIR`) if they are introduced as part of server configuration.
- [x] ~~**Basic Server Functionality Test:** Added `--smoke-test` in `check_mcp.py` to build the StreamableHTTP ASGI app and verify MCP server bootstrap wiring without binding network ports.~~

### 4.2. Usage & Integration
- [x] ~~**Add Docstring/Usage Instructions:** `check_mcp.py` now exposes documented CLI options (`--help`) for JSON output, smoke test, strict mode, and expected env checks.~~
- [x] ~~**Integrate into Development Workflow:** Added `check_mcp` coverage into the automated pytest path (`eidos_mcp/tests/test_check_mcp.py`) and documented strict preflight invocation in README.~~
- [x] ~~**Structured Output:** `check_mcp.py --json` emits a structured payload including tool/resource registries and component check results.~~

---

## 5. MCP Tooling Enhancements and Discoverability

This section outlines enhancements to the existing toolset and mechanisms for improved tool discoverability and utility for agents.

- [x] **Implement Tool Discovery Tool**: Create a new MCP tool, `mcp_list_tools`, that returns a JSON-formatted list of all registered tools, including their names, descriptions, and parameter schemas. This will enable agents to programmatically discover and understand available tools.
- [x] **Implement LLM Text Generation Tool**: Expose a direct tool, `llm_generate_text`, that allows agents to perform general-purpose text generation using `LLMForge`. This would accept a prompt and return the generated text.
- [x] **Implement File System Write Tool**: Create an MCP tool, `file_write`, to write content to a specified file. This tool should include parameters for `file_path` and `content`.
- [x] **Implement File System Read Tool**: Create an MCP tool, `file_read`, to read the content of a specified file. This tool should include a parameter for `file_path` and return the file's content.
- [x] **Implement File System Create Tool**: Create an MCP tool, `file_create`, to create a new, empty file at a specified path. This tool should include a parameter for `file_path`.
- [x] **Implement File System Delete Tool**: Create an MCP tool, `file_delete`, to delete a specified file or empty directory. This tool should include a parameter for `file_path`.
- [x] **Implement Shell Command Execution Tool**: Create an MCP tool, `run_shell_command`, that executes arbitrary shell commands. This tool should include parameters for `command` and optionally `cwd` (current working directory), and return `stdout`, `stderr`, and `exit_code`. *Note: This tool should be implemented with extreme caution due to security implications and require explicit user confirmation or a robust sandbox if exposed broadly.*
- [x] **Implement Test Execution Tool**: Create an MCP tool, `run_tests`, that can execute project-specific tests. This tool should accept parameters like `test_command` (e.g., `pytest`, `npm test`) and `test_path`, returning the test results.
- [x] **Implement Virtual Environment Run Tool**: Create an MCP tool, `venv_run`, that executes a given command within a specified Python virtual environment. This tool should accept parameters like `venv_path` and `command`.
