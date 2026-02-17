import json
from pathlib import Path

def generate_report_content(tree_items, indent_level=0):
    """Generates markdown content for a list of tree items."""
    content = ""
    indent_str = "  " * indent_level

    # Sort children for consistent output
    sorted_children = sorted(tree_items, key=lambda x: (x["type"], x["name"]))

    for item in sorted_children:
        if item["type"] == "file":
            content += f"{indent_str}- {item['name']} (File, {item['size_human']}"
            if item.get("lines_of_code", 0) > 0:
                content += f", {item['lines_of_code']} LOC"
            if item.get("word_count", 0) > 0:
                content += f", {item['word_count']} words"
            content += ")\n"
        elif item["type"] == "dir":
            content += f"{indent_str}- {item['name']}/ (Dir, {item['total_size_human']}, {item['item_count']} items"
            if item.get("total_lines_of_code", 0) > 0:
                content += f", {item['total_lines_of_code']} LOC"
            if item.get("total_word_count", 0) > 0:
                content += f", {item['total_word_count']} words"
            content += ")\n"
            if "children" in item and item["children"]:
                content += generate_report_content(item["children"], indent_level + 1)
    return content

def generate_forge_reports(report_json_path, output_dir):
    """Generates individual Markdown reports for each forge."""
    with open(report_json_path, 'r') as f:
        all_forges_data = json.load(f)

    output_dir.mkdir(parents=True, exist_ok=True)

    for forge_data in all_forges_data:
        forge_name = forge_data["name"]
        report_path = output_dir / f"{forge_name}.md"
        
        print(f"Generating report for {forge_name} to {report_path}...")

        report_content = f"# Forge Analysis Report: {forge_name}\n\n"
        report_content += "## Summary Statistics\n"
        report_content += f"- **Total Size**: {forge_data['total_size_human']}\n"
        report_content += f"- **Total Files & Dirs**: {forge_data['total_files']} items\n" # This is total_items from analyze_path, which counts files and dirs at the top level
        if forge_data.get('total_loc', 0) > 0: # Use .get with default for robustness
            report_content += f"- **Total Lines of Code**: {forge_data['total_loc']}\n"
        if forge_data.get('total_word_count', 0) > 0: # Use .get with default for robustness
            report_content += f"- **Total Word Count**: {forge_data['total_word_count']}\n"
        report_content += f"- **Path**: `{forge_data['path']}`\n\n"

        report_content += "## File and Folder Tree\n"
        # The forge_data["tree"] is already the list of top-level children
        report_content += generate_report_content(forge_data["tree"], indent_level=0)

        with open(report_path, "w") as f:
            f.write(report_content)
    
    print("\nAll forge reports generated successfully.")

if __name__ == "__main__":
    base_directory = Path(__file__).parent.parent
    report_json_path = base_directory / "reports" / "forge_analysis_report.json"
    output_reports_dir = base_directory / "reports" / "forges"
    
    generate_forge_reports(report_json_path, output_reports_dir)
