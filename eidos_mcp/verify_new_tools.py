import sys
from pathlib import Path

# Add project src to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT.parent / "prompt_forge/src"))
sys.path.append(str(ROOT.parent / "article_forge/src"))
sys.path.append(str(ROOT.parent / "glyph_forge/src"))

from eidos_mcp.forge_manager import manager
from eidos_mcp.core import list_tool_metadata

# Force initialization
manager.initialize_all(force_reload=True)

# List tools and search for the new ones
tools = [t["name"] for t in list_tool_metadata()]
new_tools = [t for t in tools if any(prefix in t for prefix in ["prompt_", "article_", "figlet_", "glyph_"])]

print(f"Total Tools in Registry: {len(tools)}")
print(f"Newly Registered Tools: {new_tools}")

# Verify specific critical tools
expected = [
    "prompt_get", "prompt_list", "prompt_save",
    "article_markdown_to_html", "article_convert_file",
    "figlet_generate", "figlet_header",
    "glyph_text_to_banner", "glyph_image_to_ascii"
]

missing = [e for e in expected if e not in tools]
if missing:
    print(f"CRITICAL: Missing expected tools: {missing}")
else:
    print("SUCCESS: All new forge tools are present in the registry!")
