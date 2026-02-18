
import sys
import os
import json
from pathlib import Path

# Setup path to find eidos_mcp
sys.path.append(os.path.abspath("eidosian_forge/eidos_mcp/src"))

# Mock environment
os.environ["EIDOS_FORGE_DIR"] = os.path.abspath("eidosian_forge")
os.environ["EIDOS_ROOT_DIR"] = os.getcwd()

try:
    from eidos_mcp.routers.refactor import refactor_analyze
    from eidos_mcp.routers.tiered_memory import eidos_remember_self
    print("Imports successful.")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_refactor():
    print("\n--- Testing refactor_analyze ---")
    # Test relative path from root
    target = "eidosian_forge/eidos_mcp/src/eidos_mcp/eidos_mcp_server.py"
    result = refactor_analyze(target)
    if "Error" in result and "file_info" not in result:
        print(f"FAILED: {result}")
    else:
        print("SUCCESS: Analysis result returned.")
        # Optional: parse json to check
        try:
            data = json.loads(result)
            print(f"File info: {data.get('file_info', {}).get('path')}")
        except:
            pass

def test_memory():
    print("\n--- Testing eidos_remember_self ---")
    # Test without 'importance'
    try:
        # Note: We can't easily test this without a running chroma/memory backend available 
        # to the script, but if the import works and function runs until it hits DB, 
        # we know the signature is correct.
        # However, since we are running this script in the same env, we might be able to.
        # But 'eidos_remember_self' calls '_get_tiered_memory' which loads from disk.
        
        result = eidos_remember_self(content="Verification test", tags=["test"])
        if "Error" in result and "Tiered memory system not available" not in result:
             print(f"FAILED: {result}")
        elif "Tiered memory system not available" in result:
             print("SKIPPED: Tiered Memory not available in script env (expected).")
        else:
             print(f"SUCCESS: {result}")
    except TypeError as e:
        print(f"FAILED: TypeError: {e}")
    except Exception as e:
        print(f"FAILED: Exception: {e}")

if __name__ == "__main__":
    test_refactor()
    test_memory()
