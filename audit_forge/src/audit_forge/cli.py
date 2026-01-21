import typer
from rich.console import Console
from pathlib import Path
from audit_forge.audit_core import AuditForge

app = typer.Typer()
console = Console()

# Default to current directory if not specified
DEFAULT_ROOT = Path.cwd()

@app.command()
def coverage(root: Path = DEFAULT_ROOT):
    """Check audit coverage."""
    audit = AuditForge(data_dir=root / "audit_data")
    stats = audit.verify_coverage(str(root))
    
    console.print(f"[bold]Audit Coverage for {root}[/bold]")
    console.print(f"Unreviewed Files: [red]{stats['unreviewed_count']}[/red]")
    if stats['unreviewed_sample']:
        console.print("\nSample Unreviewed:")
        for f in stats['unreviewed_sample']:
            console.print(f" - {f}")

@app.command()
def mark(path: str, agent: str = "user"):
    """Mark a file as reviewed."""
    root = Path.cwd()
    audit = AuditForge(data_dir=root / "audit_data")
    audit.coverage.mark_reviewed(path, agent_id=agent)
    console.print(f"[green]Marked {path} as reviewed by {agent}[/green]")

@app.command()
def todo(task: str, section: str = "Immediate"):
    """Add a todo item."""
    root = Path.cwd()
    audit = AuditForge(data_dir=root / "audit_data")
    if audit.todo_manager.add_task(section, task):
        console.print(f"[green]Added task to {section}[/green]")
    else:
        console.print(f"[yellow]Task already exists[/yellow]")

def main():
    app()
