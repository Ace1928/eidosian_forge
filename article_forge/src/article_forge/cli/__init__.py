"""
Article_forge CLI - Minimal CLI interface.
"""
import argparse
import sys
from typing import Optional, List

from article_forge.publish import convert_markdown_file


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="article-forge",
        description="Article_forge - Forge component",
    )
    parser.add_argument("--version", action="store_true", help="Show version")
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    subparsers.add_parser("status", help="Show forge status")
    publish_parser = subparsers.add_parser("publish", help="Convert Markdown to HTML/PDF")
    publish_parser.add_argument("source", help="Path to markdown source file")
    publish_parser.add_argument("--html-out", help="Output HTML file path")
    publish_parser.add_argument("--pdf-out", help="Output PDF file path (requires weasyprint)")
    
    args = parser.parse_args(argv)
    
    if args.version:
        print("Article_forge v0.1.0")
        return 0
    
    if args.command == "status":
        print("Article_forge Status: operational")
        return 0

    if args.command == "publish":
        try:
            result = convert_markdown_file(
                args.source,
                html_out=args.html_out,
                pdf_out=args.pdf_out,
            )
        except Exception as exc:
            print(f"Publish failed: {exc}")
            return 1
        if result.html_path:
            print(f"HTML written: {result.html_path}")
        if result.pdf_path:
            print(f"PDF written: {result.pdf_path}")
        return 0

    parser.print_help()
    return 0

app = main

if __name__ == "__main__":
    sys.exit(main())
