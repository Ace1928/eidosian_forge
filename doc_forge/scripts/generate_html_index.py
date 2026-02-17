#!/usr/bin/env python3
import os
from pathlib import Path

# Configuration
ROOT_DIR = Path.cwd()
FINAL_DOCS_DIR = ROOT_DIR / 'doc_forge' / 'final_docs'
INDEX_FILE = ROOT_DIR / 'index.html'

def get_tree(base_path):
    tree = {}
    for root, dirs, files in os.walk(base_path):
        rel_root = Path(root).relative_to(base_path)
        node = tree
        for part in rel_root.parts:
            node = node.setdefault(part, {})
        
        for file in files:
            if file.endswith('.md'):
                node[file] = str(rel_root / file)
    return tree

def build_html_list(tree, base_url='doc_forge/final_docs'):
    html = "<ul>"
    for key in sorted(tree.keys()):
        val = tree[key]
        if isinstance(val, dict):
            html += f"<li><strong>{key}</strong>{build_html_list(val, base_url)}</li>"
        else:
            url = f"{base_url}/{val}"
            html += f"<li><a href='{url}'>{key}</a></li>"
    html += "</ul>"
    return html

def main():
    if not FINAL_DOCS_DIR.exists():
        print("Final documentation directory not found.")
        return
        
    tree = get_tree(FINAL_DOCS_DIR)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Eidosian Forge Documentation Index</title>
    <style>
        body {{
            font-family: 'Courier New', Courier, monospace;
            background-color: #0d1117;
            color: #c9d1d9;
            padding: 20px;
        }}
        h1 {{ color: #58a6ff; }}
        a {{ color: #58a6ff; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
        ul {{ list-style-type: none; }}
        li {{ margin: 5px 0; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        .header {{ border-bottom: 2px solid #30363d; padding-bottom: 10px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💎 Eidosian Forge Documentation Index</h1>
            <p>Recursive, self-optimizing documentation system.</p>
        </div>
        <div class="content">
            {build_html_list(tree)}
        </div>
    </div>
</body>
</html>
"""
    
    with open(INDEX_FILE, 'w') as f:
        f.write(html_content)
    print(f"Index successfully generated at {INDEX_FILE}")

if __name__ == "__main__":
    main()
