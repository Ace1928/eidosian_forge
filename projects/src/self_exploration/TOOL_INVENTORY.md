# 🔧 Eidosian MCP Complete Tool Inventory

**Generated**: 2026-01-23T08:51:49.670084+00:00
**Total Tools**: 79
- Core MCP Tools: 58
- Plugin Tools: 21

---

## 📋 Core MCP Tools (58)

### Memory Router (13 tools)

| Tool | Description |
|------|-------------|
| `memory_add` | Add a new memory to episodic storage. |
| `memory_add_semantic` | Add a semantic memory entry to the knowledge memory store. |
| `memory_clear` | Clear the episodic memory store. |
| `memory_clear_semantic` | Clear the semantic memory store. |
| `memory_delete` | Delete a memory item by id. |
| `memory_delete_semantic` | Delete a semantic memory entry by id. |
| `memory_restore` | Restore episodic memory from a snapshot transaction. |
| `memory_restore_semantic` | Restore semantic memory from a snapshot transaction. |
| `memory_retrieve` | Retrieve memories relevant to a query. |
| `memory_search` | Semantic search over memory. |
| `memory_snapshot` | Create a snapshot of the episodic memory store. |
| `memory_snapshot_semantic` | Create a snapshot of the semantic memory store. |
| `memory_stats` | Return memory store statistics. |

### Knowledge Router (9 tools)

| Tool | Description |
|------|-------------|
| `grag_index` | Run a GraphRAG incremental index over scan roots. |
| `grag_query` | Query GraphRAG. |
| `grag_query_local` | Run a local GraphRAG query. |
| `kb_add` | Add fact to the knowledge graph. |
| `kb_delete` | Delete a knowledge node by id. |
| `kb_get_by_tag` | Find knowledge nodes by tag. |
| `kb_link` | Create a bidirectional link between two knowledge nodes. |
| `kb_restore` | Restore knowledge base from a transaction snapshot. |
| `kb_search` | Search the knowledge graph for matching content. |

### System Router (13 tools)

| Tool | Description |
|------|-------------|
| `file_create` | Create an empty file at the specified path. |
| `file_delete` | Delete a file or an empty directory. |
| `file_find_duplicates` | Find duplicate files by content hash. |
| `file_read` | Read a file from disk. |
| `file_restore` | Restore a file or directory from the latest transaction or a specified transa... |
| `file_search` | Search file contents for a string pattern. |
| `file_write` | Write content to a file. |
| `run_shell_command` | Execute a shell command and return stdout, stderr, and exit code. |
| `run_tests` | Execute a test command and return stdout, stderr, and exit code. |
| `system_info` | Get system details. |
| `transaction_list` | List recent transactional snapshots. |
| `transaction_restore` | Restore a transaction snapshot by id. |
| `venv_run` | Execute a command inside a Python virtual environment. |

### Gis Router (4 tools)

| Tool | Description |
|------|-------------|
| `gis_get` | Retrieve a configuration value from GIS. |
| `gis_restore` | Restore the GIS persistence store from a snapshot. |
| `gis_set` | Set a configuration value in GIS. |
| `gis_snapshot` | Create a snapshot of the GIS persistence store. |

### Audit Router (2 tools)

| Tool | Description |
|------|-------------|
| `audit_add_todo` | Append a TODO item to the system TODO list. |
| `audit_mark_reviewed` | Mark a path as reviewed in the audit coverage map. |

### Diagnostics Router (2 tools)

| Tool | Description |
|------|-------------|
| `diagnostics_metrics` | Return diagnostics metrics summary. |
| `diagnostics_ping` | Return basic diagnostics status. |

### Refactor Router (1 tools)

| Tool | Description |
|------|-------------|
| `refactor_analyze` | Analyze a Python file for structural boundaries and dependencies. |

### Auth Router (1 tools)

| Tool | Description |
|------|-------------|
| `auth_whoami` | Returns the current authentication status and identity. |

### Nexus Router (2 tools)

| Tool | Description |
|------|-------------|
| `agent_run_task` | Delegate a complex objective to the Agent Forge. Returns a plan or execution ... |
| `mcp_self_upgrade` | Upgrade the MCP server (restart service) after verifying tests. Requires a st... |

### Types Router (4 tools)

| Tool | Description |
|------|-------------|
| `type_register` | Register or update a schema in Type Forge. |
| `type_restore_snapshot` | Restore registered schemas from the latest snapshot. |
| `type_snapshot` | Create a snapshot of registered schemas. |
| `type_validate` | Validate data against a registered schema. |

### Plugins Router (7 tools)

| Tool | Description |
|------|-------------|
| `plugin_discover` | Discover and load any new plugins. |
| `plugin_list` | List all loaded plugins with their status and tool counts. |
| `plugin_reload` | Reload a plugin to pick up changes. |
| `plugin_stats` | Get detailed statistics about the plugin system. |
| `tool_info` | Get detailed information about a specific tool. |
| `tool_invoke` | Call a tool by name with arguments (for dynamic tool invocation). |
| `tool_list` | List all available tools across all plugins. |

---

## 📦 Plugin Tools (21)

### computer_control (v1.0.0) - 6 tools

| Tool | Description |
|------|-------------|
| `control_click` | Click at screen coordinates.
    
    Args:
        x: X coordinate
        y... |
| `control_emergency_stop` | Activate the emergency stop (kill switch).
    Creates the kill file to halt ... |
| `control_move_mouse` | Move mouse to coordinates.
    
    Args:
        x: X coordinate
        y: ... |
| `control_screenshot` | Capture a screenshot.
    
    Args:
        region: Optional (x, y, width, h... |
| `control_status` | Get control system status including kill-switch state.
    
    Returns:
    ... |
| `control_type_text` | Type text using the keyboard.
    
    Args:
        text: Text to type
     ... |

### self_exploration (v0.5.0) - 4 tools

| Tool | Description |
|------|-------------|
| `identity_evolve` | Create a new identity version with updated learnings |
| `identity_status` | Get current identity model status and metrics |
| `introspect` | Run a structured introspection experiment with full provenance |
| `provenance_audit` | Audit provenance records to extract patterns and improvement opportunities |

### task_automation (v1.0.0) - 6 tools

| Tool | Description |
|------|-------------|
| `task_cancel` | Cancel a queued or scheduled task.
    
    Args:
        task_id: ID of the ... |
| `task_execute` | Execute a specific task immediately.
    
    Args:
        task_id: ID of th... |
| `task_queue_add` | Add a task to the execution queue.
    
    Args:
        command: Shell comm... |
| `task_queue_list` | List tasks in the queue.
    
    Args:
        status: Filter by status (que... |
| `task_queue_status` | Get overall task queue status.
    
    Returns:
        JSON string with que... |
| `task_schedule` | Schedule a task to run after a delay.
    
    Args:
        command: Shell c... |

### web_tools (v1.0.0) - 5 tools

| Tool | Description |
|------|-------------|
| `web_download` | Download a file from URL.
    
    Args:
        url: URL to download
       ... |
| `web_extract_links` | Extract all links from a webpage.
    
    Args:
        url: URL to extract ... |
| `web_fetch` | Fetch content from a URL.
    
    Args:
        url: URL to fetch
        ti... |
| `web_hash_content` | Compute hash of content at URL without storing it.
    
    Args:
        url... |
| `web_parse_document` | Parse a document using Apache Tika.
    
    Args:
        source: URL or fil... |

---

## 🏗️ Architecture

```
eidosian_forge/eidos_mcp/
├── src/eidos_mcp/
│   ├── core.py              # Tool registration decorator
│   ├── eidos_mcp_server.py  # Server entry point
│   ├── routers/             # Core tool implementations
│   │   ├── memory.py        # Episodic & semantic memory
│   │   ├── knowledge.py     # Knowledge graph & GraphRAG
│   │   ├── system.py        # File ops, shell, venv
│   │   ├── gis.py           # Global Information Store
│   │   ├── audit.py         # Audit tracking
│   │   ├── diagnostics.py   # Health & metrics
│   │   ├── refactor.py      # Code analysis
│   │   ├── auth.py          # Authentication
│   │   ├── nexus.py         # Agent & self-upgrade
│   │   ├── types.py         # Schema validation
│   │   └── plugins.py       # Plugin management
│   └── plugins/             # Dynamic plugins
│       ├── __init__.py      # Plugin loader
│       ├── self_exploration/
│       ├── computer_control/
│       ├── web_tools/
│       └── task_automation/
```

---

## 📊 Summary

| Category | Count |
|----------|-------|
| Core Routers | 11 |
| Core Tools | 58 |
| Plugins | 4 |
| Plugin Tools | 21 |
| **Total Tools** | **79** |

---

**Fully Eidosian. Always Evolving.**

