import importlib.machinery
import importlib.util
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


pdf_tools_menu = load_module("pdf_tools_menu", ROOT / "pdf_tools_menu")


class DummyTk:
    def title(self, _):
        return None

    def mainloop(self):
        return None


class DummyButton:
    def __init__(self, _root, text=None, command=None):
        self.command = command

    def pack(self, **_kwargs):
        if self.command:
            self.command()


def test_run_script_missing(tmp_path):
    pdf_tools_menu.SCRIPT_DIR = tmp_path
    assert pdf_tools_menu.run_script("missing", []) == 1


def test_run_script_ok(tmp_path, monkeypatch):
    script = tmp_path / "pdf_check_text"
    script.write_text("#!/usr/bin/env python3\nprint('ok')\n")
    pdf_tools_menu.SCRIPT_DIR = tmp_path
    monkeypatch.setattr(pdf_tools_menu.subprocess, "call", lambda *a, **k: 0)
    assert pdf_tools_menu.run_script("pdf_check_text", []) == 0


def test_cli_main(monkeypatch):
    monkeypatch.setattr(pdf_tools_menu, "run_script", lambda *a, **k: 0)
    monkeypatch.setattr(
        pdf_tools_menu.sys,
        "argv",
        ["pdf_tools_menu", "--check-text", "--file", "f.pdf"],
    )
    assert pdf_tools_menu.cli_main() == 0

    monkeypatch.setattr(
        pdf_tools_menu.sys,
        "argv",
        ["pdf_tools_menu", "--to-text", "--file", "f.pdf", "--output", "o.txt"],
    )
    assert pdf_tools_menu.cli_main() == 0

    monkeypatch.setattr(
        pdf_tools_menu.sys,
        "argv",
        ["pdf_tools_menu", "--check-encryption", "--file", "f.pdf"],
    )
    assert pdf_tools_menu.cli_main() == 0

    monkeypatch.setattr(pdf_tools_menu, "gui_main", lambda: 0)
    monkeypatch.setattr(pdf_tools_menu.sys, "argv", ["pdf_tools_menu"])
    assert pdf_tools_menu.cli_main() == 0


def test_gui_main(monkeypatch):
    monkeypatch.setattr(pdf_tools_menu.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_tools_menu.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_tools_menu, "run_script", lambda *a, **k: 0)
    assert pdf_tools_menu.gui_main() == 0


def test_pdf_tools_menu_entrypoint():
    try:
        runpy.run_path(str(ROOT / "pdf_tools_menu"), run_name="__main__")
    except SystemExit:
        pass
