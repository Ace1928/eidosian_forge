import importlib.machinery
import importlib.util
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


pdf_check_text = load_module("pdf_check_text", ROOT / "pdf_check_text")
pdf_check_encryption = load_module("pdf_check_encryption", ROOT / "pdf_check_encryption")
pdf_to_text = load_module("pdf_to_text", ROOT / "pdf_to_text")


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


class DummyDialog:
    def __init__(self, open_value=None, save_value=None):
        self.open_value = open_value
        self.save_value = save_value

    def askopenfilename(self, **_kwargs):
        return self.open_value or ""

    def asksaveasfilename(self, **_kwargs):
        return self.save_value or ""


class DummyMessageBox:
    def showinfo(self, *_args, **_kwargs):
        return None

    def showwarning(self, *_args, **_kwargs):
        return None

    def showerror(self, *_args, **_kwargs):
        return None


def test_check_pdf_text(monkeypatch):
    class Result:
        returncode = 0
        stdout = "hello"
        stderr = ""

    monkeypatch.setattr(pdf_check_text, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_text.subprocess, "run", lambda *a, **k: Result())
    assert pdf_check_text.check_pdf_text("file.pdf") is True


def test_check_pdf_text_error(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "fail"

    monkeypatch.setattr(pdf_check_text, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_text.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError):
        pdf_check_text.check_pdf_text("file.pdf")


def test_pdf_check_text_cli(monkeypatch, capsys):
    monkeypatch.setattr(pdf_check_text, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_text, "check_pdf_text", lambda _: True)
    monkeypatch.setattr(pdf_check_text, "gui_main", lambda: 0)
    monkeypatch.setattr(pdf_check_text.sys, "argv", ["pdf_check_text", "--file", "file.pdf"])
    assert pdf_check_text.cli_main() == 0
    assert "has_text" in capsys.readouterr().out

    monkeypatch.setattr(pdf_check_text.sys, "argv", ["pdf_check_text", "--file", "file.pdf", "--json"])
    assert pdf_check_text.cli_main() == 0

    def raise_err(_):
        raise RuntimeError("fail")

    monkeypatch.setattr(pdf_check_text, "check_pdf_text", raise_err)
    monkeypatch.setattr(pdf_check_text.sys, "argv", ["pdf_check_text", "--file", "file.pdf", "--json"])
    assert pdf_check_text.cli_main() == 1

    monkeypatch.setattr(pdf_check_text, "check_pdf_text", raise_err)
    monkeypatch.setattr(pdf_check_text.sys, "argv", ["pdf_check_text", "--file", "file.pdf"])
    assert pdf_check_text.cli_main() == 1

    monkeypatch.setattr(pdf_check_text.sys, "argv", ["pdf_check_text"])
    assert pdf_check_text.cli_main() == 0


def test_pdf_check_text_gui(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf")
    monkeypatch.setattr(pdf_check_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_text, "messagebox", DummyMessageBox())
    monkeypatch.setattr(pdf_check_text, "check_pdf_text", lambda _: True)
    assert pdf_check_text.gui_main() == 0


def test_pdf_check_text_gui_no_text(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf")
    monkeypatch.setattr(pdf_check_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_text, "messagebox", DummyMessageBox())
    monkeypatch.setattr(pdf_check_text, "check_pdf_text", lambda _: False)
    assert pdf_check_text.gui_main() == 0


def test_pdf_check_text_gui_no_file(monkeypatch):
    dialog = DummyDialog(open_value="")
    monkeypatch.setattr(pdf_check_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_text, "messagebox", DummyMessageBox())
    assert pdf_check_text.gui_main() == 0


def test_pdf_check_text_gui_error(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf")
    monkeypatch.setattr(pdf_check_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_text, "messagebox", DummyMessageBox())
    monkeypatch.setattr(
        pdf_check_text,
        "check_pdf_text",
        lambda _: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    assert pdf_check_text.gui_main() == 0


def test_is_encrypted(monkeypatch):
    class Result:
        returncode = 1
        stdout = ""
        stderr = "Password required"

    monkeypatch.setattr(pdf_check_encryption, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_encryption.subprocess, "run", lambda *a, **k: Result())
    assert pdf_check_encryption.is_encrypted("file.pdf") is True


def test_is_not_encrypted(monkeypatch):
    class Result:
        returncode = 0
        stdout = "ok"
        stderr = ""

    monkeypatch.setattr(pdf_check_encryption, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_encryption.subprocess, "run", lambda *a, **k: Result())
    assert pdf_check_encryption.is_encrypted("file.pdf") is False


def test_pdf_check_encryption_cli(monkeypatch):
    monkeypatch.setattr(pdf_check_encryption, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_check_encryption, "is_encrypted", lambda _: False)
    monkeypatch.setattr(pdf_check_encryption, "gui_main", lambda: 0)
    monkeypatch.setattr(pdf_check_encryption.sys, "argv", ["pdf_check_encryption", "--file", "file.pdf"])
    assert pdf_check_encryption.cli_main() == 0

    monkeypatch.setattr(
        pdf_check_encryption.sys,
        "argv",
        ["pdf_check_encryption", "--file", "file.pdf", "--json"],
    )
    assert pdf_check_encryption.cli_main() == 0

    def raise_err(_):
        raise RuntimeError("fail")

    monkeypatch.setattr(pdf_check_encryption, "is_encrypted", raise_err)
    monkeypatch.setattr(
        pdf_check_encryption.sys,
        "argv",
        ["pdf_check_encryption", "--file", "file.pdf", "--json"],
    )
    assert pdf_check_encryption.cli_main() == 1

    monkeypatch.setattr(pdf_check_encryption.sys, "argv", ["pdf_check_encryption", "--file", "file.pdf"])
    assert pdf_check_encryption.cli_main() == 1

    monkeypatch.setattr(pdf_check_encryption.sys, "argv", ["pdf_check_encryption"])
    assert pdf_check_encryption.cli_main() == 0


def test_pdf_check_encryption_gui(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf")
    monkeypatch.setattr(pdf_check_encryption.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_encryption.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_encryption, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_encryption, "messagebox", DummyMessageBox())
    monkeypatch.setattr(pdf_check_encryption, "is_encrypted", lambda _: False)
    assert pdf_check_encryption.gui_main() == 0


def test_pdf_check_encryption_gui_encrypted(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf")
    monkeypatch.setattr(pdf_check_encryption.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_encryption.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_encryption, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_encryption, "messagebox", DummyMessageBox())
    monkeypatch.setattr(pdf_check_encryption, "is_encrypted", lambda _: True)
    assert pdf_check_encryption.gui_main() == 0


def test_pdf_check_encryption_gui_no_file(monkeypatch):
    dialog = DummyDialog(open_value="")
    monkeypatch.setattr(pdf_check_encryption.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_check_encryption.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_check_encryption, "filedialog", dialog)
    monkeypatch.setattr(pdf_check_encryption, "messagebox", DummyMessageBox())
    assert pdf_check_encryption.gui_main() == 0


def test_convert_pdf_overwrite(tmp_path, monkeypatch):
    pdf = tmp_path / "input.pdf"
    pdf.write_text("data")
    out = tmp_path / "out.txt"

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    def fake_run(cmd, capture_output=False, text=False):
        Path(cmd[-1]).write_text("converted")
        return Result()

    monkeypatch.setattr(pdf_to_text.subprocess, "run", fake_run)
    pdf_to_text.convert_pdf(str(pdf), str(out), overwrite=True)
    assert out.exists()


def test_convert_pdf_create_file(tmp_path, monkeypatch):
    pdf = tmp_path / "input.pdf"
    pdf.write_text("data")
    out = tmp_path / "out.txt"

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(pdf_to_text.subprocess, "run", lambda *a, **k: Result())
    pdf_to_text.convert_pdf(str(pdf), str(out), overwrite=False)
    assert out.exists()


def test_convert_pdf_exists_no_overwrite(tmp_path, monkeypatch):
    pdf = tmp_path / "input.pdf"
    pdf.write_text("data")
    out = tmp_path / "out.txt"
    out.write_text("existing")

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(pdf_to_text.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError):
        pdf_to_text.convert_pdf(str(pdf), str(out), overwrite=False)


def test_convert_pdf_error(tmp_path, monkeypatch):
    pdf = tmp_path / "input.pdf"
    pdf.write_text("data")
    out = tmp_path / "out.txt"

    class Result:
        returncode = 1
        stdout = ""
        stderr = "fail"

    monkeypatch.setattr(pdf_to_text.subprocess, "run", lambda *a, **k: Result())
    with pytest.raises(RuntimeError):
        pdf_to_text.convert_pdf(str(pdf), str(out), overwrite=True)


def test_pdf_to_text_cli(monkeypatch):
    monkeypatch.setattr(pdf_to_text, "require_cmd", lambda _: None)
    monkeypatch.setattr(pdf_to_text, "convert_pdf", lambda *a, **k: None)
    monkeypatch.setattr(pdf_to_text, "gui_main", lambda: 0)

    monkeypatch.setattr(
        pdf_to_text.sys,
        "argv",
        ["pdf_to_text", "--input", "in.pdf", "--output", "out.txt"],
    )
    assert pdf_to_text.cli_main() == 0

    monkeypatch.setattr(
        pdf_to_text.sys,
        "argv",
        ["pdf_to_text", "--input", "in.pdf", "--output", "out.txt", "--json"],
    )
    assert pdf_to_text.cli_main() == 0

    def raise_err(*_args, **_kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(pdf_to_text, "convert_pdf", raise_err)
    monkeypatch.setattr(
        pdf_to_text.sys,
        "argv",
        ["pdf_to_text", "--input", "in.pdf", "--output", "out.txt", "--json"],
    )
    assert pdf_to_text.cli_main() == 1

    monkeypatch.setattr(
        pdf_to_text.sys,
        "argv",
        ["pdf_to_text", "--input", "in.pdf", "--output", "out.txt"],
    )
    assert pdf_to_text.cli_main() == 1

    monkeypatch.setattr(pdf_to_text.sys, "argv", ["pdf_to_text"])
    assert pdf_to_text.cli_main() == 0


def test_pdf_to_text_gui(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf", save_value="out.txt")
    monkeypatch.setattr(pdf_to_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_to_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_to_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_to_text, "messagebox", DummyMessageBox())
    monkeypatch.setattr(pdf_to_text, "convert_pdf", lambda *a, **k: None)
    assert pdf_to_text.gui_main() == 0


def test_pdf_to_text_gui_no_file(monkeypatch):
    dialog = DummyDialog(open_value="")
    monkeypatch.setattr(pdf_to_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_to_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_to_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_to_text, "messagebox", DummyMessageBox())
    assert pdf_to_text.gui_main() == 0


def test_pdf_to_text_gui_no_save(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf", save_value="")
    monkeypatch.setattr(pdf_to_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_to_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_to_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_to_text, "messagebox", DummyMessageBox())
    assert pdf_to_text.gui_main() == 0


def test_pdf_to_text_gui_error(monkeypatch):
    dialog = DummyDialog(open_value="file.pdf", save_value="out.txt")
    monkeypatch.setattr(pdf_to_text.tk, "Tk", DummyTk)
    monkeypatch.setattr(pdf_to_text.tk, "Button", DummyButton)
    monkeypatch.setattr(pdf_to_text, "filedialog", dialog)
    monkeypatch.setattr(pdf_to_text, "messagebox", DummyMessageBox())
    monkeypatch.setattr(
        pdf_to_text,
        "convert_pdf",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("fail")),
    )
    assert pdf_to_text.gui_main() == 0


def test_pdf_main_entrypoints():
    for script in ("pdf_check_text", "pdf_check_encryption", "pdf_to_text"):
        path = ROOT / script
        try:
            runpy.run_path(str(path), run_name="__main__")
        except SystemExit:
            pass
