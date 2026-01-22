import os
import sys
import importlib.util
from pathlib import Path

ROOT_DIR = Path("eidosian_forge").resolve()
EXCLUDE_DIRS = {
    "data", "docs", "requirements", "scripts", "audit_data", "__pycache__", ".git", ".vscode", "projects", "node_modules"
}

def get_forges():
    return [
        d for d in os.listdir(ROOT_DIR)
        if (ROOT_DIR / d).is_dir() and d not in EXCLUDE_DIRS and not d.startswith(".")
    ]

def check_forge(forge_name):
    forge_path = ROOT_DIR / forge_name
    src_path = forge_path / "src"
    package_path = src_path / forge_name
    
    issues = []
    
    # 1. Structure Check
    if not src_path.exists():
        issues.append("Missing 'src' directory")
    elif not package_path.exists():
        # Handle case where package name might differ (e.g., dashes vs underscores)
        # But per standard, it should be eidosian_forge/<name>/src/<name>
        # Let's check if *any* package exists in src
        subdirs = [d for d in os.listdir(src_path) if (src_path / d).is_dir() and not d.startswith(".")]
        if not subdirs:
             issues.append(f"No package directory found in 'src'")
        else:
             # Just note if it doesn't match exactly, might be okay but worth noting
             if forge_name not in subdirs and forge_name.replace("-", "_") not in subdirs:
                 issues.append(f"Package name mismatch in src: found {subdirs}")

    # 2. Config Check
    has_pyproject = (forge_path / "pyproject.toml").exists()
    has_setup = (forge_path / "setup.py").exists()
    
    if not (has_pyproject or has_setup):
        issues.append("Missing build config (pyproject.toml or setup.py)")

    # 3. Import Check
    if src_path.exists():
        # Try to find the importable name
        import_name = forge_name.replace("-", "_")
        # If the dir in src is different, use that
        if (src_path / import_name).exists():
            pass
        else:
             # look for likely candidate
             candidates = [d for d in os.listdir(src_path) if (src_path / d).is_dir()]
             if candidates:
                 import_name = candidates[0]

        try:
            # We don't want to actually import everything (might run code), just find spec
            # But "functional as a module" implies importability.
            # Let's try basic import spec.
            sys.path.insert(0, str(src_path))
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                issues.append(f"Cannot find spec for module '{import_name}'")
            sys.path.pop(0)
        except Exception as e:
            issues.append(f"Import check error: {e}")

    return issues

def audit_projects():
    projects_dir = ROOT_DIR / "projects" / "src"
    if not projects_dir.exists():
        return {"projects": ["Missing projects/src directory"]}
    
    results = {}
    if not projects_dir.exists():
        return results

    for proj in os.listdir(projects_dir):
        proj_path = projects_dir / proj
        if proj_path.is_dir() and not proj.startswith("."):
            issues = []
            # Projects might not follow the strict src/<name> pattern inside projects/src/<name>
            # The instruction was: eidosian_forge/projects/src/<project_name>
            # So the project root IS <project_name>. Does it have source inside? 
            # Usually projects have their own src or just files. 
            # Let's just check if they look like python projects.
            
            # Check for pyproject or setup
            has_config = (proj_path / "pyproject.toml").exists() or (proj_path / "setup.py").exists()
            if not has_config:
                 issues.append("Missing build config")
            
            # Check for some code structure (basic check)
            if not any(proj_path.glob("*.py")) and not (proj_path / "src").exists() and not any(proj_path.glob("*/*.py")):
                issues.append("No Python files or src dir found")
                
            if issues:
                results[f"project/{proj}"] = issues
    return results

def main():
    print(f"Auditing forges in {ROOT_DIR}...")
    forges = get_forges()
    forges.sort()
    
    results = {}
    for forge in forges:
        res = check_forge(forge)
        if res:
            results[forge] = res
            
    proj_results = audit_projects()
    results.update(proj_results)

    if not results:
        print("ALL CLEAR: All forges appear structurally correct.")
    else:
        print("\nISSUES FOUND:")
        for forge, issues in results.items():
            print(f"\n[{forge}]")
            for issue in issues:
                print(f"  - {issue}")

if __name__ == "__main__":
    main()
