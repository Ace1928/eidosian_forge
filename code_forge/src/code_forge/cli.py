import typer
from pathlib import Path
from code_forge.analyzer.python_analyzer import CodeAnalyzer
from code_forge.librarian.core import CodeLibrarian
from rich.console import Console

app = typer.Typer()
console = Console()

@app.command()
def analyze(path: Path):
    """Analyze a Python file."""
    analyzer = CodeAnalyzer()
    if path.is_file():
        result = analyzer.analyze_file(path)
        console.print(result)
    else:
        console.print("[red]Path must be a file.[/red]")

@app.command()
def ingest(path: Path, lib_path: Path = Path("./data/code_library.json")):
    """Ingest a file into the Code Library."""
    analyzer = CodeAnalyzer()
    librarian = CodeLibrarian(lib_path)
    
    if not path.is_file():
        return

    analysis = analyzer.analyze_file(path)
    content = path.read_text(encoding="utf-8")
    
    sid = librarian.add_snippet(content, metadata=analysis)
    console.print(f"[green]Ingested {path.name} as {sid[:8]}[/green]")

def main():
    app()
