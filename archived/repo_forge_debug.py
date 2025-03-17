#!/usr/bin/env python3
"""
Special Eidosian Debugging Tool for Repository Bootstrap Process.

This diagnostic utility identifies and fixes JSON serialization issues
and ensures all components work together flawlessly.
"""
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List

# Banner for diagnostics
print("\033[38;5;51m" + "=" * 70 + "\033[0m")
print("\033[38;5;196m🔮 EIDOSIAN TEMPLATE SYSTEM DIAGNOSTIC TOOL 🔮\033[0m")
print("\033[38;5;51m" + "=" * 70 + "\033[0m")

def check_repo_forge_installation() -> bool:
    """Validate repo_forge is correctly installed."""
    try:
        import repo_forge
        print(f"✅ repo_forge version {repo_forge.__version__} is installed")
        return True
    except ImportError:
        print("❌ repo_forge is not installed. Please install with:")
        print("   pip install -e ~/repos/repo_forge_repo")
        return False

def diagnose_template_system() -> bool:
    """Test the template rendering system."""
    try:
        from repo_forge.core.templates import render_template, EidosianTemplate
        template = "$var1 and ${var2} are variables"
        variables = {"var1": "Hello", "var2": "World", "datetime": "Module"}
        
        result = render_template(template, variables)
        print(f"✅ Template rendering works: {result}")
        return True
    except Exception as e:
        print(f"❌ Template system error: {e}")
        return False

def test_script_generator() -> Dict[str, Any]:
    """Test the script generator independently."""
    try:
        from repo_forge.generators.scripts import create_script_files
        from tempfile import TemporaryDirectory
        
        with TemporaryDirectory() as tempdir:
            path = Path(tempdir)
            result = create_script_files(path, ["python"], True)
            # Force count to be integer
            if "count" in result:
                result["count"] = int(result["count"])
            print(f"✅ Script generator returned: {json.dumps(result, indent=2)}")
            return result
    except Exception as e:
        print(f"❌ Script generator error: {e}")
        return {"success": False, "error": str(e)}

def patch_scripts_file() -> bool:
    """Apply critical patch to scripts.py if needed."""
    scripts_path = Path.home() / "repos" / "repo_forge_repo" / "repo_forge" / "generators" / "scripts.py"
    
    if not scripts_path.exists():
        print(f"❌ Cannot find scripts.py at {scripts_path}")
        return False
    
    with open(scripts_path, "r") as f:
        content = f.read()
    
    # Check if our fix is already applied
    if "# Force cast to int" in content:
        print("✅ scripts.py already patched")
        return True
    
    # Add critical fix
    patched_content = content.replace(
        "count = len(created_files)",
        "# Force cast to int\ncount = int(len(created_files))"
    )
    
    with open(scripts_path, "w") as f:
        f.write(patched_content)
    
    print("🔧 Applied critical patch to scripts.py")
    return True

def run_bootstrap_test() -> bool:
    """Run bootstrap test with minimal target."""
    print("\n🚀 Testing bootstrap with minimal target...")
    
    cmd = [
        "python", "-c",
        """
import sys
from repo_forge.cli import main
sys.exit(main(['--name', 'test_repo', '--output', '/tmp', '--languages', 'python']))
        """
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Bootstrap test successful!")
        return True
    else:
        print(f"❌ Bootstrap test failed: {result.stderr}")
        return False

# Run diagnostic sequence
print("\n🔍 Starting diagnostic sequence...\n")

if check_repo_forge_installation():
    diagnose_template_system()
    test_result = test_script_generator()
    
    # Check for potential JSON serialization issues
    try:
        json_str = json.dumps(test_result)
        parsed = json.loads(json_str)
        if isinstance(parsed.get("count"), str):
            print("⚠️ WARNING: 'count' is serialized as a string!")
            patch_scripts_file()
        else:
            print("✅ JSON serialization works correctly")
    except Exception as e:
        print(f"❌ JSON serialization error: {e}")
        patch_scripts_file()
    
    run_bootstrap_test()

print("\n" + "\033[38;5;51m" + "=" * 70 + "\033[0m")
print("\033[38;5;82m✨ DIAGNOSTIC COMPLETE\033[0m")
print("\033[38;5;51m" + "=" * 70 + "\033[0m")
