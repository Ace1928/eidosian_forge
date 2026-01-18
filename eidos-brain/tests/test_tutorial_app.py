from unittest.mock import patch
from pathlib import Path
import subprocess
import sys
from labs import tutorial_app


def test_main_exits_quickly():
    with patch("rich.prompt.Prompt.ask", side_effect=["exit"]):
        tutorial_app.main()


def test_save_and_load_memory(tmp_path: Path) -> None:
    """Save a memory then load it back and confirm output."""
    memory_file = tmp_path / "mem.txt"
    with patch("rich.prompt.Prompt.ask", side_effect=["add", "hello", "exit"]), patch(
        "rich.console.Console.print"
    ) as mock_print:
        tutorial_app.main(save=str(memory_file))
        outputs = "".join(call.args[0] for call in mock_print.call_args_list)
    assert memory_file.exists()
    assert outputs.count("Memories saved") == 1
    with patch("rich.prompt.Prompt.ask", side_effect=["reflect", "exit"]), patch(
        "rich.console.Console.print"
    ) as mock_print:
        tutorial_app.main(load=str(memory_file))
        prints = "".join(call.args[0] for call in mock_print.call_args_list)
    assert "Loaded 1 memories" in prints


def test_save_called_once(tmp_path: Path) -> None:
    """``save_memory`` should be invoked exactly once on exit."""
    memory_file = tmp_path / "mem.txt"
    with patch("labs.tutorial_app.save_memory") as mock_save, patch(
        "rich.prompt.Prompt.ask", side_effect=["exit"]
    ):
        tutorial_app.main(save=str(memory_file))
    assert mock_save.call_count == 1


def test_parser_arguments() -> None:
    """Parser should accept ``--load`` and ``--save``."""
    parser = tutorial_app.build_parser()
    args = parser.parse_args(["--load", "in.txt", "--save", "out.txt"])
    assert args.load == "in.txt"
    assert args.save == "out.txt"


def test_cli_help():
    result = subprocess.run(
        [
            sys.executable,
            "labs/tutorial_app.py",
            "--help",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Eidos interactive tutorial" in result.stdout


def test_save_memory_creates_file(tmp_path: Path) -> None:
    """Ensure ``save_memory`` writes data to disk."""
    core = tutorial_app.EidosCore()
    core.remember("note")
    file_path = tmp_path / "saved.txt"
    console = tutorial_app.Console(record=True)
    tutorial_app.save_memory(core, file_path, console)
    assert file_path.exists()
    assert file_path.read_text() == "note"
    assert "Memories saved" in console.export_text()


def test_load_memory_existing_file(tmp_path: Path) -> None:
    """Verify ``load_memory`` populates core from an existing file."""
    file_path = tmp_path / "loaded.txt"
    file_path.write_text("a\nb\n")
    core = tutorial_app.EidosCore()
    console = tutorial_app.Console(record=True)
    tutorial_app.load_memory(core, file_path, console)
    assert core.memory == ["a", "b"]
    assert "Loaded 2 memories" in console.export_text()


def test_recursion_after_load(tmp_path: Path) -> None:
    """Ensure recursion continues to work after loading memories."""
    file_path = tmp_path / "recurse.txt"
    file_path.write_text("hello")
    core = tutorial_app.EidosCore()
    tutorial_app.load_memory(core, file_path, tutorial_app.Console())
    core.recurse()
    assert len(core.memory) == 2
    assert any(isinstance(m, dict) for m in core.memory)
