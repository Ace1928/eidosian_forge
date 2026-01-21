import os
import subprocess
import shutil
from pathlib import Path
import tempfile

# Configuration
GITHUB_USER = "Ace1928"
FORGE_ROOT = Path("/home/lloyd/eidosian_forge")
IGNORE_PATTERNS = shutil.ignore_patterns(".git", "__pycache__", ".venv", "venv", "*.pyc")

def sync_forge(forge_name):
    repo_url = f"https://github.com/{GITHUB_USER}/{forge_name}.git"
    target_dir = FORGE_ROOT / forge_name
    
    print(f"🔄 Processing {forge_name}...")
    
    # Check if remote exists
    try:
        subprocess.run(
            ["git", "ls-remote", repo_url, "HEAD"], 
            check=True, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"⚠️  Remote repo not found for {forge_name}. Skipping.")
        return False

    # Clone to temp
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"⬇️  Cloning {repo_url}...")
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, temp_dir],
                check=True,
                stdout=subprocess.DEVNULL
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to clone {forge_name}: {e}")
            return False

        # Sync files
        if not target_dir.exists():
            print(f"🆕 Creating local directory for {forge_name}")
            target_dir.mkdir(parents=True, exist_ok=True)
            
        print(f"📂 Syncing content to {target_dir}...")
        
        # We use copytree with dirs_exist_ok=True to overlay
        # We must manually iterate to avoid 'shutil.Error: Destination path already exists' in older python or just strict overlay
        # Actually shutil.copytree in Python 3.8+ has dirs_exist_ok
        try:
            shutil.copytree(temp_dir, target_dir, dirs_exist_ok=True, ignore=IGNORE_PATTERNS)
            print(f"✅ Synced {forge_name}")
            return True
        except Exception as e:
            print(f"❌ Failed to copy files for {forge_name}: {e}")
            return False

def main():
    # Identify forge directories
    forges = [
        d.name for d in FORGE_ROOT.iterdir() 
        if d.is_dir() and (d.name.endswith("_forge") or d.name in ["eidos_brain", "graphrag"])
    ]
    
    # Sort for cleaner logs
    forges.sort()
    
    success_count = 0
    fail_count = 0
    
    print(f"🔍 Found {len(forges)} potential sub-forges.")
    
    for forge in forges:
        if sync_forge(forge):
            success_count += 1
        else:
            fail_count += 1
            
    print(f"\n🏁 Sync Complete. Success: {success_count}, Skipped/Failed: {fail_count}")

if __name__ == "__main__":
    main()
