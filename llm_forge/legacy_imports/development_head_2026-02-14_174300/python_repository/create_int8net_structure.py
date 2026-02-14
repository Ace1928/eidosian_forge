"""
Script: manage_int8net_structure.py

ADVANCED directory-structure management script for the "int8net" project.
Now includes:
  1) Configurable base_dir, project_name, templates_dir from JSON.
  2) Extended MIME and extension maps merged with built-in defaults.
  3) Dynamic 'templates' creation and usage: if a file references a template,
     the script attempts to copy that content into the new file. If missing, 
     falls back to a minimal placeholder.
  4) Preserves all fancy printing, color codes, icons, stylized directory 
     tree generation, listing, previewing, and user confirmation features.

Usage examples remain the same:
  python manage_int8net_structure.py
  python manage_int8net_structure.py --preview
  python manage_int8net_structure.py --create
  python manage_int8net_structure.py --list
  python manage_int8net_structure.py --confirm
  etc.

Requires Python 3.6+.
"""

import os
import json
import argparse
import mimetypes
from pathlib import Path

# ---------------------------------------------------------------------
# Global / Default Config
# ---------------------------------------------------------------------

# If you rename your JSON, change here or override with CLI argument if needed.
CONFIG_FILE = "extended_structure_config.json"
OUTPUT_STRUCTURE_FILE = "int8net_directory_structure.txt"

# ANSI color/style codes for fancy logs
COLOR_RESET = "\033[0m"
COLOR_BOLD = "\033[1m"
COLOR_BLUE = "\033[94m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_RED = "\033[91m"
COLOR_CYAN = "\033[96m"
COLOR_MAGENTA = "\033[95m"

# Emojis / Icons (pick your favorites!)
ICON_INFO = "ℹ️ "
ICON_WARN = "⚠️ "
ICON_ERROR = "❌"
ICON_SUCCESS = "✅"
ICON_FOLDER = "📂"
ICON_FILE_GENERIC = "📄"

# Built-in MIME icons (we merge with user-provided).
DEFAULT_MIME_ICON_MAP = {
      "application/x-tar": "📦🗜️",
      "application/x-zip-compressed": "📂🗜️", 
      "application/x-rar-compressed": "📚🗜️",
      "application/x-7z-compressed": "📁🗜️",
      "application/gzip": "📦🌀",
      "application/x-bzip2": "📂🌀",
      "application/pdf": "📕📄",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "📝✍️",
      "application/msword": "📄✍️",
      "application/vnd.ms-powerpoint": "🎯📊",
      "application/vnd.openxmlformats-officedocument.presentationml.presentation": "🎭📊",
      "application/vnd.ms-excel": "📊📈",
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "💹📊",
      "application/json": "📋🗄️",
      "application/xml": "📜🗄️",
      "application/javascript": "💻📜",
      "application/x-python-code": "🐍💻",
      "application/x-shellscript": "🐚💻",
      "application/octet-stream": "💾📦",
      "application/x-msdownload": "💻📥",
      "application/x-dosexec": "⚡💻",
      "application/rtf": "📄📝",
      "application/x-httpd-php": "🌐💻",
      "application/x-shockwave-flash": "🎬⚡",
      "application/x-java-archive": "☕📦",
      "application/x-font-ttf": "🔤📝",
      "application/x-font-woff": "🔤💫",
      "application/x-font-woff2": "🔤✨",
      "application/x-font-opentype": "🔤🎨",
      "application/x-font-truetype": "🔤📋",
      "application/x-font-bdf": "🔤🖨️",
      "application/x-font-pcf": "🔤🗜️",
      "application/postscript": "📜⚡",
      "application/wasm": "⚡🔧",
      "application/x-sqlite3": "🗄️💾",
      "application/x-sql": "🗄️📝",
      "application/x-iso9660-image": "💿📀",
      "application/x-cue": "💿🎵",
      "application/x-vnd.oasis.opendocument.text": "📄📝",
      "application/x-vnd.oasis.opendocument.spreadsheet": "📊📈",
      "application/x-vnd.oasis.opendocument.presentation": "🎯📊",
      "text/plain": "📄📝",
      "text/markdown": "📝📚",
      "text/html": "🌐📄",
      "text/css": "🎨💅",
      "text/csv": "📊📋",
      "text/tab-separated-values": "📋📊",
      "text/x-python": "🐍📜",
      "text/x-shellscript": "🐚📜",
      "text/x-java-source": "☕📜",
      "text/x-csrc": "💻📟",
      "text/x-c++src": "💻⚡",
      "text/x-csharp": "🎯💻",
      "text/x-rustsrc": "🦀💻",
      "text/x-go": "🐹💻",
      "text/x-kotlin": "🤖💻",
      "text/x-perl": "🐪💻",
      "text/x-ruby": "💎💻",
      "text/x-php": "🌐💻",
      "text/x-lua": "🌙💻",
      "text/x-sql": "🗄️📝",
      "text/x-json": "📋🗄️",
      "image/jpeg": "📸🖼️",
      "image/png": "🎨🖼️",
      "image/gif": "🎞️🖼️",
      "image/svg+xml": "🎨📐",
      "image/tiff": "📷📊",
      "image/x-icon": "🎨📍",
      "image/x-ms-bmp": "🖼️💾",
      "audio/mpeg": "🎵🎧",
      "audio/ogg": "🎶🎵",
      "audio/wav": "🔊🎼",
      "audio/x-flac": "🎼🎵",
      "audio/mp4": "🎵📱",
      "audio/x-ms-wma": "🎵💿",
      "video/mp4": "🎥📱",
      "video/x-matroska": "🎬📀",
      "video/x-msvideo": "📽️💿",
      "video/quicktime": "🎥🍎",
      "video/x-ms-wmv": "🎬💿",
      "video/x-flv": "🎬💻",
      "video/x-mpeg2": "🎬📺",
      "application/vnd.android.package-archive": "📱📦",
      "application/x-apple-diskimage": "🍎💿",
      "application/x-diskcopy-image": "💿📀",
      "application/x-iso-image": "💿📀",
      "application/vnd.debian.binary-package": "🐧📦",
      "application/x-redhat-package-manager": "🎩📦",
      "application/x-ms-installer": "🪟📦",
      "application/vnd.mozilla.xul+xml": "🦊🌐",
      "application/x-dockerfile": "🐳📄",
      "application/x-makefile": "🔧📄",
      "text/x-makefile": "🔧📋",
      "text/x-dockerfile": "🐳📋",
      "text/x-jenkinsfile": "🛠️📋",
      "text/x-markdown-template": "📝✨",
      "text/x-template": "📋✨",
      "text/x-gitignore": "🚫📋",
      "text/x-license": "⚖️📜",
      "application/x-config": "⚙️📋",
      "text/x-readme": "ℹ️📚",
      "application/x-certificate": "🔐📜",
      "text/x-toml": "⚙️📝",
      "text/x-yaml": "⚙️📋",
      "application/x-python-template": "🐍✨",
      "text/x-shell-template": "🐚✨",
      "text/x-properties": "⚙️📝"
  }

# Built-in extension icons (merged with user-provided).
DEFAULT_EXTENSION_ICON_MAP = {
        ".txt": "📄📝",
        ".cfg": "⚙️📋",
        ".log": "📊📝",
        ".ini": "⚙️📄",
        ".md": "📝📚",
        ".html": "🌐📄",
        ".css": "🎨🖌️",
        ".csv": "📊📈",
        ".tsv": "📊📉",
        ".py": "🐍📜",
        ".sh": "🐚📜",
        ".js": "💻📜",
        ".json": "📋🗒️",
        ".xml": "📜📑",
        ".php": "🌐💻",
        ".java": "☕📜",
        ".c": "💻📟",
        ".cpp": "💻⚡",
        ".cs": "🎯💻",
        ".rs": "🦀💻",
        ".go": "🐹💻",
        ".kt": "🤖💻",
        ".rb": "💎💻",
        ".pl": "🐪💻",
        ".lua": "🌙💻",
        ".sql": "🗄️💾",
        ".yml": "📋🔧",
        ".yaml": "📋⚙️",
        ".svg": "🎨📐",
        ".jpg": "📷🖼️",
        ".jpeg": "📸🖼️",
        ".png": "🎨🖼️",
        ".gif": "🎞️🎭",
        ".tiff": "📷📊",
        ".bmp": "🖼️💾",
        ".ico": "🎨📍",
        ".mp3": "🎵🎧",
        ".wav": "🔊🎼",
        ".flac": "🎼🎵",
        ".ogg": "🎶🎵",
        ".mp4": "🎥📹",
        ".mkv": "🎬📹",
        ".avi": "📽️🎬",
        ".mov": "🎥🎬",
        ".wmv": "📹🎬",
        ".flv": "🎬💻",
        ".tar": "📦🗜️",
        ".zip": "📂🗜️",
        ".rar": "🗜️📚",
        ".gz": "🗜️📦",
        ".bz2": "🗜️📂",
        ".7z": "🗜️📁",
        ".iso": "💿📀",
        ".dmg": "🍎📀",
        ".deb": "🐧📦",
        ".rpm": "🎩📦",
        ".msi": "🪟📦",
        ".apk": "📱📦",
        ".exe": "⚡💻",
        ".bin": "💾💻",
        ".pdf": "📕📄",
        ".doc": "📝📄",
        ".docx": "📄✍️",
        ".xls": "📊📈",
        ".xlsx": "📊💹",
        ".ppt": "📊🎯",
        ".pptx": "📊🎭",
        ".odt": "📄📝",
        ".ods": "📊📈",
        ".odp": "📊🎯",
        ".sqlite3": "🗄️💾",
        ".db": "🗄️📁",
        ".wasm": "⚡🔧",
        ".ttf": "🔤📝",
        ".woff": "🔤💫",
        ".woff2": "🔤✨",
        ".ps": "📜⚡",
        ".jar": "☕📦",
        ".xul": "🦊🌐",
        ".dockerfile": "🐳📄",
        ".toml": "⚙️📝",
        ".tpl": "📑✨",
        ".makefile": "🔧📄",
        ".jenkinsfile": "🛠️📄",
        ".license": "📜⚖️",
        ".gitignore": "🚫📄",
        ".readme": "📚ℹ️",
        ".crt": "🔐📜",
        ".pem": "🔑📜",
        ".conf": "⚙️📄",
        ".properties": "⚙️📝",
        ".md.tpl": "📝✨",
        ".py.tpl": "🐍✨",
        ".sh.tpl": "🐚✨",
        ".json.tpl": "📋✨",
        ".html.tpl": "🌐✨"
    }

# ---------------------------------------------------------------------
# Utility / Fancy Logging
# ---------------------------------------------------------------------

def fancy_print(msg: str, color_code: str = COLOR_RESET, icon: str = ""):
    """Print a message with fancy styling."""
    print(f"{color_code}{icon}{msg}{COLOR_RESET}")


# ---------------------------------------------------------------------
# Main Logic
# ---------------------------------------------------------------------

def load_config(json_file: str) -> dict:
    """
    Load the entire advanced config from JSON.
    Should contain keys: 'base_dir', 'project_name', 'templates_dir',
    'mime_map', 'extension_map', and 'structure'.
    """
    if not Path(json_file).exists():
        raise FileNotFoundError(
            f"{ICON_ERROR} Configuration file '{json_file}' not found in current directory."
        )

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Basic check
    required_keys = ["base_dir", "project_name", "templates_dir", "structure"]
    for rk in required_keys:
        if rk not in data:
            raise ValueError(f"Missing required key '{rk}' in config.")

    return data


def merge_mime_maps(user_mime_map: dict, user_ext_map: dict) -> None:
    """
    Merge user-provided MIME / extension maps into our defaults.
    (Modifies the global dicts in place, or returns new dicts if you prefer.)
    """
    # Merge MIME
    for k, v in user_mime_map.items():
        DEFAULT_MIME_ICON_MAP[k] = v

    # Merge extension
    for k, v in user_ext_map.items():
        DEFAULT_EXTENSION_ICON_MAP[k] = v


def guess_file_icon(file_path: Path) -> str:
    """
    Attempt to guess the best icon for a file based on its MIME type or extension.
    Uses the merged MIME/extension maps.
    """
    mime_type, _ = mimetypes.guess_type(str(file_path))
    if mime_type and (mime_type in DEFAULT_MIME_ICON_MAP):
        return DEFAULT_MIME_ICON_MAP[mime_type]

    # If not found in mime map, try extension map
    if file_path.suffix.lower() in DEFAULT_EXTENSION_ICON_MAP:
        return DEFAULT_EXTENSION_ICON_MAP[file_path.suffix.lower()]

    return ICON_FILE_GENERIC


def setup_templates_dir(base_path: Path, templates_dir: str, template_list: list):
    """
    Ensure the templates directory exists and create any inline-defined templates
    as files if they're not already present.
    """
    template_dir_path = base_path / templates_dir
    if not template_dir_path.exists():
        fancy_print(f"Creating templates directory: {template_dir_path}", color_code=COLOR_BLUE, icon=ICON_FOLDER + " ")
        template_dir_path.mkdir(parents=True, exist_ok=True)
    else:
        fancy_print(f"Templates directory already exists: {template_dir_path}", color_code=COLOR_CYAN, icon=ICON_FOLDER + " ")

    # Create each template file if it doesn't exist
    for t in template_list:
        tpl_name = t.get("filename", "")
        tpl_content = t.get("content", "")
        if not tpl_name:
            continue

        dest_file = template_dir_path / tpl_name
        if not dest_file.exists():
            fancy_print(f"Creating template file: {dest_file}", color_code=COLOR_BLUE, icon=guess_file_icon(dest_file) + " ")
            with dest_file.open('w', encoding='utf-8') as f:
                f.write(tpl_content)
        else:
            fancy_print(f"Template file exists: {dest_file} (skipping)", color_code=COLOR_CYAN, icon=guess_file_icon(dest_file) + " ")


def load_template_contents(templates_dir_path: Path, template_name: str) -> str:
    """
    Load the template content from file if it exists, else return an empty string.
    """
    file_path = templates_dir_path / template_name
    if file_path.exists():
        return file_path.read_text(encoding='utf-8')
    else:
        fancy_print(
            f"Template '{template_name}' not found under '{templates_dir_path}'. Using a minimal placeholder.",
            color_code=COLOR_YELLOW,
            icon=ICON_WARN + " "
        )
        return ""


def create_structure(base_path: Path,
                     templates_dir: str,
                     structure_dirs: dict):
    """
    Create the directories/files as defined in the advanced structure config.
    - 'structure_dirs' is the dictionary of directories -> file arrays.
    - Each file array element has 'filename' and optionally 'template'.
    - If 'template' is specified, we try to load it from templates_dir.
    - If the directory is outside 'project_name', we skip it.
    """
    # We expect to place everything under base_path (which is base_dir/project_name)
    # We'll also need the templates dir path
    template_dir_path = base_path / templates_dir

    for directory, file_entries in structure_dirs.items():
        # The user can define an empty directory key -> means the root project folder
        target_dir = base_path / directory

        # Create directory if needed
        if not target_dir.exists():
            fancy_print(f"Creating directory: {target_dir}", color_code=COLOR_BLUE, icon=ICON_FOLDER + " ")
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            fancy_print(f"Directory already exists: {target_dir}", color_code=COLOR_CYAN, icon=ICON_FOLDER + " ")

        # For each file
        for entry in file_entries:
            filename = entry.get("filename", "")
            template_name = entry.get("template", "")
            if not filename:
                continue  # skip empty

            file_path = target_dir / filename
            if file_path.exists():
                fancy_print(f"File already exists: {file_path} (skipping)", color_code=COLOR_CYAN, icon=guess_file_icon(file_path) + " ")
                continue

            # Create the file
            fancy_print(f"Creating file: {file_path}", color_code=COLOR_BLUE, icon=guess_file_icon(file_path) + " ")
            file_path.touch()

            # If a template is specified, load and write it
            if template_name:
                tpl_content = load_template_contents(template_dir_path, template_name)
                # Optionally, we can do some token substitution, e.g. {FILENAME}:
                tpl_content = tpl_content.replace("{FILENAME}", filename)
                file_path.write_text(tpl_content, encoding='utf-8')
            else:
                # If no template is specified, or doesn't exist, just a minimal placeholder:
                if file_path.suffix == ".py":
                    file_path.write_text(
                        f"# {filename}\n\n# Placeholder for {filename}\n",
                        encoding='utf-8'
                    )


# ---------------------------------------------------------------------
# Listing / Preview / Confirmation / Tree
# ---------------------------------------------------------------------

def list_structure(base_dir: Path, project_name: str):
    """
    Lists the directory structure under 'base_dir' (plus project_name) with fancy styling,
    showing icons based on MIME type or extension for each file.
    """
    project_path = base_dir / project_name
    fancy_print(f"\nListing directory structure under: {project_path}", color_code=COLOR_BOLD, icon=ICON_INFO + " ")
    if not project_path.exists():
        fancy_print(
            f"Project directory does not exist yet: {project_path}",
            color_code=COLOR_YELLOW,
            icon=ICON_WARN + " "
        )
        return

    for root, dirs, files in os.walk(project_path):
        level = root.replace(str(project_path), '').count(os.sep)
        indent = ' ' * 4 * level
        folder_name = Path(root).name
        fancy_print(f"{indent}{folder_name}/", color_code=COLOR_GREEN, icon=ICON_FOLDER + " ")
        sub_indent = ' ' * 4 * (level + 1)
        for file_name in files:
            file_path = Path(root) / file_name
            file_icon = guess_file_icon(file_path)
            fancy_print(f"{sub_indent}{file_name}", color_code=COLOR_RESET, icon=file_icon + " ")


def show_update_preview(base_path: Path,
                        structure_dirs: dict,
                        templates_dir: str):
    """
    Show a preview of what will be created vs. what already exists,
    skipping existing items.
    """
    fancy_print("\nPreview of changes (directories/files to be created):", color_code=COLOR_BOLD, icon=ICON_INFO + " ")
    template_dir_path = base_path / templates_dir

    for directory, file_entries in structure_dirs.items():
        target_dir = base_path / directory
        if not target_dir.exists():
            fancy_print(f"  Will create directory: {target_dir}", color_code=COLOR_BLUE, icon=ICON_FOLDER + " ")

        for entry in file_entries:
            filename = entry.get("filename", "")
            if not filename:
                continue
            file_path = target_dir / filename
            if not file_path.exists():
                fancy_print(
                    f"    Will create file: {file_path}",
                    color_code=COLOR_BLUE,
                    icon=guess_file_icon(file_path) + " "
                )
            else:
                fancy_print(
                    f"    File already exists: {file_path} (no action)",
                    color_code=COLOR_CYAN,
                    icon=guess_file_icon(file_path) + " "
                )


def confirm_action(prompt: str, required_confirmation: str = "CONFIRM") -> bool:
    """
    Repeatedly ask for confirmation (up to 3 tries).
    Returns True if user types 'CONFIRM' exactly, otherwise False.
    """
    fancy_print(prompt, color_code=COLOR_BOLD, icon=ICON_INFO + " ")
    for _ in range(3):
        user_input = input(f"Type '{required_confirmation}' to confirm: ").strip()
        if user_input == required_confirmation:
            return True
    return False


def generate_directory_tree(base_path: Path,
                            project_name: str,
                            output_file: Path):
    """
    Generates a stylized directory tree in ASCII-art style
    under base_path/project_name and writes it to output_file.
    """
    fancy_print(f"Writing a stylized directory tree to: {output_file}", color_code=COLOR_MAGENTA, icon=ICON_INFO + " ")

    project_path = base_path / project_name
    if not project_path.exists():
        fancy_print("Project path does not exist yet. Skipping tree generation.", color_code=COLOR_RED, icon=ICON_WARN + " ")
        return

    lines = []

    def walk_and_build(current_dir: Path, depth: int = 0):
        indent = " " * (4 * depth)
        lines.append(f"{indent}{ICON_FOLDER} {current_dir.name}/")

        subfolders = sorted([d for d in current_dir.iterdir() if d.is_dir()])
        regular_files = sorted([f for f in current_dir.iterdir() if f.is_file()])

        # Files first, then subfolders
        for f in regular_files:
            file_icon = guess_file_icon(f)
            lines.append(f"{indent}    {file_icon} {f.name}")

        for folder in subfolders:
            walk_and_build(folder, depth + 1)

    walk_and_build(project_path)

    with output_file.open('w', encoding='utf-8') as f:
        f.write("========================================\n")
        f.write(f"  {project_name} Project Directory Tree\n")
        f.write("========================================\n\n")
        f.write(f"Base Directory: {project_path.as_posix()}\n\n")
        for line in lines:
            f.write(line + "\n")

    fancy_print(f"Directory tree written to: {output_file}", color_code=COLOR_GREEN, icon=ICON_SUCCESS + " ")


# ---------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Advanced creation/management for the 'int8net' project directory structure."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_FILE,
        help=f"Path to the extended structure config JSON (default: {CONFIG_FILE})."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list the directory structure under base_dir/project_name."
    )
    parser.add_argument(
        "--create",
        action="store_true",
        help="Create/verify the project directory structure."
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a preview of what would be created, but do nothing else."
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Prompt user confirmation before creating files/folders."
    )

    args = parser.parse_args()

    # 1) Load the extended config
    try:
        config_data = load_config(args.config)
    except Exception as e:
        fancy_print(
            f"Failed to load extended config from '{args.config}': {e}",
            color_code=COLOR_RED,
            icon=ICON_ERROR + " "
        )
        exit(1)

    # 2) Merge user-provided MIME and extension maps
    user_mime_map = config_data.get("mime_map", {})
    user_ext_map = config_data.get("extension_map", {})
    merge_mime_maps(user_mime_map, user_ext_map)

    # 3) Identify top-level directories
    base_dir = Path(config_data["base_dir"]).resolve()
    project_name = config_data["project_name"]
    templates_dir = config_data["templates_dir"]

    # 4) structure info
    structure_obj = config_data["structure"]
    template_list = structure_obj.get("templates", [])
    dirs_obj = structure_obj.get("dirs", {})

    # 5) Derive final project path, ensure it exists or will exist
    project_path = base_dir / project_name

    # 6) Create / verify the templates directory and template files
    setup_templates_dir(project_path, templates_dir, template_list)

    # 7) If user just wants to list
    if args.list:
        list_structure(base_dir, project_name)
        # Also generate tree
        tree_file = project_path / OUTPUT_STRUCTURE_FILE
        generate_directory_tree(base_dir, project_name, tree_file)
        exit(0)

    # 8) If user wants a preview
    if args.preview:
        show_update_preview(project_path, dirs_obj, templates_dir)
        # Also generate the text file for the current existing structure
        tree_file = project_path / OUTPUT_STRUCTURE_FILE
        generate_directory_tree(base_dir, project_name, tree_file)
        exit(0)

    # 9) Default behavior: list + create if no flags
    if not (args.create or args.preview or args.list):
        fancy_print("No action flags provided; defaulting to: list + create", color_code=COLOR_BOLD, icon=ICON_INFO + " ")
        list_structure(base_dir, project_name)
        show_update_preview(project_path, dirs_obj, templates_dir)

        if args.confirm:
            if not confirm_action("\nDo you want to proceed with creation?"):
                fancy_print("Creation canceled by user.", color_code=COLOR_YELLOW, icon=ICON_WARN + " ")
                exit(0)

        create_structure(project_path, templates_dir, dirs_obj)
        # Write final stylized tree
        tree_file = project_path / OUTPUT_STRUCTURE_FILE
        generate_directory_tree(base_dir, project_name, tree_file)
        exit(0)

    # 10) Otherwise, user explicitly wants to create
    if args.create:
        show_update_preview(project_path, dirs_obj, templates_dir)

        if args.confirm:
            if not confirm_action("\nDo you want to proceed with creation?"):
                fancy_print("Creation canceled by user.", color_code=COLOR_YELLOW, icon=ICON_WARN + " ")
                exit(0)

        create_structure(project_path, templates_dir, dirs_obj)
        # Always generate final stylized tree
        tree_file = project_path / OUTPUT_STRUCTURE_FILE
        generate_directory_tree(base_dir, project_name, tree_file)
        exit(0)
