import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add project root and src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "eidos_mcp" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "eidos_mcp" / "src"))

from eidos_mcp.forge_manager import ForgeManager

@pytest.fixture
def mock_mcp():
    return MagicMock()

@pytest.fixture
def forge_manager(mock_mcp):
    return ForgeManager(mcp_instance=mock_mcp)

def test_forge_manager_initialization(forge_manager, mock_mcp):
    assert forge_manager.mcp == mock_mcp
    assert len(forge_manager.router_modules) > 0

@patch("importlib.import_module")
def test_register_routers(mock_import, forge_manager):
    forge_manager.register_routers()
    # Verify we at least tried to import one of our routers
    assert mock_import.called
    mock_import.assert_any_call("eidos_mcp.routers.gis")

@patch("eidos_mcp.forge_manager.list_tool_metadata")
def test_sync_agent_tools(mock_list_tools, forge_manager):
    # Setup mock tools
    mock_list_tools.return_value = [
        {"name": "test_tool", "func": lambda: "ok", "description": "A test tool"}
    ]
    
    # Mock the agent
    mock_agent = MagicMock()
    forge_manager.agent = mock_agent
    
    forge_manager.sync_agent_tools()
    
    mock_agent.register_tool.assert_called_once_with(
        "test_tool", 
        mock_list_tools.return_value[0]["func"], 
        "A test tool"
    )

@patch("eidos_mcp.forge_manager.init_plugins")
@patch("eidos_mcp.forge_manager.list_tools")
def test_load_plugins(mock_list_tools, mock_init, forge_manager, mock_mcp):
    mock_init.return_value = ["plugin1"]
    mock_list_tools.return_value = ["tool1", "tool2"]
    
    forge_manager.load_plugins()
    
    mock_init.assert_called_once_with(mock_mcp)

def test_initialize_all(forge_manager):
    # This is a high-level check to ensure all sub-initializers are called
    with patch.object(forge_manager, "register_routers") as mock_reg, \
         patch.object(forge_manager, "sync_agent_tools") as mock_sync, \
         patch.object(forge_manager, "load_plugins") as mock_load:
        
        forge_manager.initialize_all()
        
        mock_reg.assert_called_once()
        mock_sync.assert_called_once()
        mock_load.assert_called_once()
