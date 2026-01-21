import sys
import os
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Add project root to path so we can find eidos_mcp and other forges
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

@pytest.fixture
def mock_dependencies():
    with patch("subprocess.Popen") as mock_popen, \
         patch("mcp.server.fastmcp.FastMCP") as mock_fastmcp:
        
        # Setup mock process for chatmock
        mock_process = MagicMock()
        mock_process.poll.return_value = None
        mock_popen.return_value = mock_process
        
        # Setup mock MCP
        mock_mcp_instance = MagicMock()
        mock_fastmcp.return_value = mock_mcp_instance
        
        # Mock the decorators to capture registered tool names
        registered_tools = []
        registered_resources = []

        def tool_decorator(name=None):
            def decorator(func):
                tool_name = name if name else func.__name__
                registered_tools.append(tool_name)
                return func
            return decorator
            
        def resource_decorator(uri):
            def decorator(func):
                registered_resources.append(uri)
                return func
            return decorator

        mock_mcp_instance.tool.side_effect = tool_decorator
        mock_mcp_instance.resource.side_effect = resource_decorator
        
        yield mock_popen, mock_fastmcp, mock_mcp_instance, registered_tools, registered_resources

def test_server_imports_and_initialization(mock_dependencies):
    mock_popen, mock_fastmcp, mock_mcp_instance, registered_tools, registered_resources = mock_dependencies
    
    # Import the server module
    # We reload it to ensure fresh execution for the test if it was already imported
    if 'eidos_mcp.eidos_mcp_server' in sys.modules:
        del sys.modules['eidos_mcp.eidos_mcp_server']
    for module_name in list(sys.modules.keys()):
        if module_name.startswith("eidos_mcp.core") or module_name.startswith("eidos_mcp.routers"):
            del sys.modules[module_name]
        
    import eidos_mcp.eidos_mcp_server as server
    
    # Verify imports worked (existence of forges)
    assert server.gis is not None
    assert server.llm is not None
    assert server.refactor is not None
    assert server.agent is not None
    
    # Verify MCP server was instantiated
    mock_fastmcp.assert_called()
    assert server.mcp == mock_mcp_instance
    
    # Verify critical tools are registered
    expected_tools = [
        "gis_get", "gis_set",
        "memory_add", "memory_retrieve",
        "kb_add", "grag_query",
        "file_read", "file_write",
        "agent_run_task", "mcp_self_upgrade"
    ]
    
    for tool in expected_tools:
        assert tool in registered_tools, f"Tool {tool} was not registered"

    # Verify critical resources are registered
    expected_resources = [
        "eidos://config",
        "eidos://persona",
        "eidos://roadmap"
    ]
    
    for res in expected_resources:
        assert res in registered_resources, f"Resource {res} was not registered"
