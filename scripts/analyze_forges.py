import os
import json
from pathlib import Path
import math # Import the math module

# --- Helper Functions ---

def get_lines_of_code(file_path):
    """Counts non-empty lines in a source file."""
    if not file_path.is_file():
        return 0
    # Heuristic for source file extensions
    source_extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.sh', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.md', '.txt', '.toml', '.ini'}
    if file_path.suffix.lower() not in source_extensions:
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for line in f if line.strip())
    except IOError:
        return 0

def get_word_count(file_path):
    """Counts words in a text file."""
    if not file_path.is_file():
        return 0
    text_extensions = {'.py', '.js', '.ts', '.java', '.c', '.cpp', '.h', '.hpp', '.go', '.rs', '.rb', '.php', '.cs', '.sh', '.json', '.yaml', '.yml', '.xml', '.html', '.css', '.md', '.txt', '.toml', '.ini', '.log', '.csv', '.tsv'}
    if file_path.suffix.lower() not in text_extensions:
        return 0
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            return len(content.split())
    except IOError:
        return 0

def get_human_readable_size(size_bytes):
    """Converts bytes to human-readable format."""
    if size_bytes == 0:
        return "0 B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024))) # Corrected: use math.floor and math.log
    p = math.pow(1024, i) # Corrected: use math.pow
    s = round(size_bytes / p, 2)
    return f"{s} {size_name[i]}"

# --- Core Analysis Logic ---

def analyze_path(current_path):
    """Recursively analyzes a path and collects metadata."""
    tree_data = []
    total_size_bytes = 0
    total_lines_of_code = 0
    total_word_count = 0
    item_count = 0

    if not current_path.exists():
        return tree_data, total_size_bytes, total_lines_of_code, total_word_count, item_count

    for entry in current_path.iterdir():
        # Skip hidden files/directories (starting with .)
        if entry.name.startswith('.'):
            continue

        item_count += 1
        entry_data = {
            "name": entry.name,
            "path": str(entry.relative_to(current_path.parent)) # Relative to forge root for cleaner paths
        }

        if entry.is_file():
            size_bytes = entry.stat().st_size
            lines_of_code = get_lines_of_code(entry)
            word_count = get_word_count(entry)
            
            entry_data.update({
                "type": "file",
                "size_bytes": size_bytes,
                "size_human": get_human_readable_size(size_bytes),
                "lines_of_code": lines_of_code,
                "word_count": word_count,
            })
            total_size_bytes += size_bytes
            total_lines_of_code += lines_of_code
            total_word_count += word_count

        elif entry.is_dir():
            sub_tree, sub_size, sub_loc, sub_wc, sub_item_count = analyze_path(entry)
            entry_data.update({
                "type": "dir",
                "total_size_bytes": sub_size,
                "total_size_human": get_human_readable_size(sub_size),
                "total_lines_of_code": sub_loc,
                "total_word_count": sub_wc,
                "item_count": sub_item_count,
                "children": sub_tree
            })
            total_size_bytes += sub_size
            total_lines_of_code += sub_loc
            total_word_count += sub_wc

        tree_data.append(entry_data)
        
    return tree_data, total_size_bytes, total_lines_of_code, total_word_count, item_count

def get_forge_directories(base_path):
    forge_dirs = []
    for entry in os.listdir(base_path):
        if os.path.isdir(os.path.join(base_path, entry)) and entry.endswith("_forge") and entry != "archive_forge":
            forge_dirs.append(Path(base_path) / entry)
    return sorted(forge_dirs)

def analyze_forges(base_path):
    forge_directories = get_forge_directories(base_path)
    analysis_results = []

    for forge_dir in forge_directories:
        forge_name = forge_dir.name
        print(f"Starting detailed analysis for {forge_name}...")
        
        # Perform recursive analysis for the current forge
        tree, total_size, total_loc, total_wc, total_items = analyze_path(forge_dir)
        
        analysis_results.append({
            "name": forge_name,
            "path": str(forge_dir),
            "total_size_bytes": total_size,
            "total_size_human": get_human_readable_size(total_size),
            "total_files": total_items, # Note: This counts files and directories at the top level
            "total_loc": total_loc,
            "total_word_count": total_wc,
            "tree": tree
        })
    
    # Sort forges by total size (smallest to largest)
    analysis_results.sort(key=lambda x: x['total_size_bytes'])

    return analysis_results

if __name__ == "__main__":
    base_directory = Path(__file__).parent.parent # /data/data/com.termux/files/home/eidosian_forge
    
    results = analyze_forges(base_directory)
    
    # Output to a JSON file
    output_path = base_directory / "reports" / "forge_analysis_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAnalysis complete. Results saved to {output_path}")