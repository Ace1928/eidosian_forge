from eidos_mcp.routers.knowledge import kb_apply_decay, kb_detect_conflicts, kb_add_fact, kb_get_by_tag
import json

def test_kb_mcp_tools():
    print("Testing kb_add_fact...")
    res = kb_add_fact("The Eidosian Forge is expanding.", ["meta"])
    print(f"Result: {res}")
    
    print("\nTesting kb_get_by_tag...")
    res = kb_get_by_tag("meta")
    print(f"Result: {res}")
    
    print("\nTesting kb_apply_decay...")
    res = kb_apply_decay(factor=0.1, threshold=0.1)
    print(f"Result: {res}")
    
    print("\nTesting kb_detect_conflicts...")
    res = kb_detect_conflicts("Eidosian Forge expansion")
    print(f"Result: {res}")

if __name__ == "__main__":
    test_kb_mcp_tools()
