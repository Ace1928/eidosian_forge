import builtins
import importlib.machinery
import importlib.util
import json
import sys
from pathlib import Path

FORGE_PATH = Path(__file__).resolve().parents[1] / "forge_builder"


def fake_rich_module():
    class DummyConsole:
        def print(self, *_args, **_kwargs):
            return None

    class DummyProgress:
        def __init__(self, *_args, **_kwargs):
            return None

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def add_task(self, *_args, **_kwargs):
            return 1

        def update(self, *_args, **_kwargs):
            return None

    class DummyColumn:
        def __init__(self, *_args, **_kwargs):
            return None

    class DummyPrompt:
        @staticmethod
        def ask(_prompt, default=None):
            return default

    class DummyConfirm:
        @staticmethod
        def ask(_prompt, default=False):
            return default

    class DummyTree:
        def __init__(self, label):
            self.label = label
            self.children = []

        def add(self, label):
            node = DummyTree(label)
            self.children.append(node)
            return node

    class DummyPanel:
        @staticmethod
        def fit(_text, **_kwargs):
            return "panel"

    class DummyTable:
        def __init__(self, **_kwargs):
            self.rows = []

        def add_column(self, *_args, **_kwargs):
            return None

        def add_row(self, *_args, **_kwargs):
            self.rows.append(_args)

    class DummyRichHandler:
        def __init__(self, **_kwargs):
            return None

    rich = type("rich", (), {})()
    rich.print = lambda *a, **k: None
    rich.console = type("console", (), {"Console": DummyConsole})
    rich.progress = type(
        "progress",
        (),
        {
            "Progress": DummyProgress,
            "SpinnerColumn": DummyColumn,
            "TextColumn": DummyColumn,
            "BarColumn": DummyColumn,
            "TimeElapsedColumn": DummyColumn,
        },
    )
    rich.prompt = type("prompt", (), {"Prompt": DummyPrompt, "Confirm": DummyConfirm})
    rich.tree = type("tree", (), {"Tree": DummyTree})
    rich.panel = type("panel", (), {"Panel": DummyPanel})
    rich.table = type("table", (), {"Table": DummyTable})
    rich.logging = type("logging", (), {"RichHandler": DummyRichHandler})
    return rich


def load_forge_builder(with_rich=True, module_name="forge_builder_test"):
    original_import = builtins.__import__
    if with_rich:
        sys.modules["rich"] = fake_rich_module()
        sys.modules["rich.console"] = sys.modules["rich"].console
        sys.modules["rich.progress"] = sys.modules["rich"].progress
        sys.modules["rich.prompt"] = sys.modules["rich"].prompt
        sys.modules["rich.tree"] = sys.modules["rich"].tree
        sys.modules["rich.panel"] = sys.modules["rich"].panel
        sys.modules["rich.table"] = sys.modules["rich"].table
        sys.modules["rich.logging"] = sys.modules["rich"].logging
    else:
        for key in list(sys.modules):
            if key.startswith("rich"):
                sys.modules.pop(key, None)

        def fake_import(name, *args, **kwargs):
            if name.startswith("rich"):
                raise ImportError("no rich")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = fake_import

    try:
        loader = importlib.machinery.SourceFileLoader(module_name, str(FORGE_PATH))
        spec = importlib.util.spec_from_loader(module_name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
        return module
    finally:
        builtins.__import__ = original_import


def test_ui_elements():
    mod = load_forge_builder(module_name="forge_builder_ui")
    console = mod.Console()
    mod.UIElements.header(console)
    mod.UIElements.section(console, "Title")
    mod.UIElements.success(console, "done")
    mod.UIElements.warning(console, "warn")
    mod.UIElements.error(console, "err")
    config = mod.ForgeConfig(name="demo")
    assert config.get_root_dir() == Path("demo")


def test_create_path_success_and_error(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_create")
    config = mod.ForgeConfig(name="demo", verbose=True)
    fs = mod.FileSystemOperator(config, mod.Console())

    dir_path = tmp_path / "alpha"
    file_path = tmp_path / "beta" / "file.txt"
    assert fs.create_path(dir_path, is_file=False) is True
    assert dir_path.exists()
    assert fs.create_path(file_path, is_file=True) is True
    assert file_path.exists()

    def bad_touch(self, *args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(Path, "touch", bad_touch)
    assert fs.create_path(tmp_path / "oops.txt", is_file=True) is False
    assert fs.operations_count["errors"] == 1


def test_safely_move_contents_success_and_error(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_move")
    config = mod.ForgeConfig(name="demo")
    fs = mod.FileSystemOperator(config, mod.Console())

    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (src / "file.txt").write_text("hi")

    fs.safely_move_contents(src, dest)
    assert (dest / "file.txt").exists()
    assert not src.exists()

    bad_src = tmp_path / "bad"
    bad_dest = tmp_path / "bad_dest"
    bad_src.mkdir()
    bad_dest.mkdir()

    def bad_iterdir(self):
        raise OSError("broken")

    monkeypatch.setattr(Path, "iterdir", bad_iterdir)
    fs.safely_move_contents(bad_src, bad_dest)
    assert fs.operations_count["errors"] >= 1


def test_safely_move_contents_skip_existing(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_move_skip")
    config = mod.ForgeConfig(name="demo")
    fs = mod.FileSystemOperator(config, mod.Console())

    src = tmp_path / "src"
    dest = tmp_path / "dest"
    src.mkdir()
    dest.mkdir()
    (src / "file.txt").write_text("hi")
    (dest / "file.txt").write_text("existing")

    fs.safely_move_contents(src, dest)
    assert fs.operations_count["skipped"] >= 1


def test_unique_temp_dir(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_temp")
    path = tmp_path / "alpha"
    path.mkdir()
    assert mod.unique_temp_dir(path) == tmp_path / "alpha_temp"

    (tmp_path / "alpha_temp").mkdir()
    assert mod.unique_temp_dir(path) == tmp_path / "alpha_temp_1"
    for idx in range(1, 100):
        (tmp_path / f"alpha_temp_{idx}").mkdir()
    try:
        mod.unique_temp_dir(path)
    except RuntimeError as exc:
        assert "unique temp directory" in str(exc)


def test_load_default_config_and_save(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_default")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    assert mod.load_default_config() == {}

    config = mod.ForgeConfig(
        name="demo", core_folders=["src"], base_files=["README.md"]
    )
    mod.save_as_default_config(config)
    saved = json.loads((tmp_path / ".eidosian" / "forge_config.json").read_text())
    assert saved["core_folders"] == ["src"]

    (tmp_path / ".eidosian" / "forge_config.json").write_text("{bad")
    assert mod.load_default_config() == {}


def test_save_default_config_error(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_save_error")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    def bad_open(*_args, **_kwargs):
        raise OSError("nope")

    monkeypatch.setattr(builtins, "open", bad_open)
    config = mod.ForgeConfig(name="demo")
    mod.save_as_default_config(config)


def test_load_config_file(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_config")
    assert mod.load_config_file(None) == {}

    missing = tmp_path / "missing.json"
    assert mod.load_config_file(missing) == {}

    bad = tmp_path / "bad.json"
    bad.write_text("{bad")
    assert mod.load_config_file(bad) == {}

    ok = tmp_path / "ok.json"
    ok.write_text(json.dumps({"core_folders": ["src"]}))
    assert mod.load_config_file(ok) == {"core_folders": ["src"]}


def test_create_structure_dry_run_rich(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_dry")
    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.DRY_RUN,
        output_dir=tmp_path,
        core_folders=["docs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert not (tmp_path / "demo").exists()


def test_create_structure_interactive_rich(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_interactive")
    root = tmp_path / "demo"
    existing = root / "docs"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old")

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.INTERACTIVE,
        output_dir=tmp_path,
        core_folders=["docs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert (root / "docs" / "old.txt").exists()
    assert (root / "README.md").exists()


def test_create_structure_interactive_skips(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_interactive_skip")
    root = tmp_path / "demo"
    root.mkdir()
    (root / "README.md").write_text("hi")
    (root / "docs").mkdir()

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.INTERACTIVE,
        output_dir=tmp_path,
        skip_existing=True,
        core_folders=["docs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert (root / "README.md").exists()


def test_create_structure_parallel(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_parallel")
    root = tmp_path / "demo"
    root.mkdir()
    (root / "docs").mkdir()
    (root / "README.md").write_text("hi")

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        parallel=True,
        skip_existing=True,
        core_folders=["docs"],
        base_files=["README.md", "new.txt"],
    )
    mod.create_structure(config)
    assert (root / "new.txt").exists()


def test_create_structure_parallel_failures(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_parallel_fail")

    def fake_create(self, path, is_file=False):
        if path.name == "fail.txt":
            return False
        if path.name == "boom.txt":
            raise RuntimeError("boom")
        return True

    monkeypatch.setattr(mod.FileSystemOperator, "create_path", fake_create)
    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        parallel=True,
        core_folders=[],
        base_files=["ok.txt", "fail.txt", "boom.txt"],
    )
    mod.create_structure(config)


def test_create_structure_sequential(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_seq")
    root = tmp_path / "demo"
    existing = root / "logs"
    existing.mkdir(parents=True)
    (existing / "old.log").write_text("old")

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        core_folders=["logs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert (root / "logs" / "old.log").exists()
    assert (root / "README.md").exists()


def test_create_structure_sequential_skip_existing(tmp_path):
    mod = load_forge_builder(module_name="forge_builder_seq_skip")
    root = tmp_path / "demo"
    root.mkdir()
    (root / "README.md").write_text("hi")
    (root / "logs").mkdir()

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        skip_existing=True,
        core_folders=["logs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert (root / "README.md").exists()


def test_create_structure_non_rich(tmp_path):
    mod = load_forge_builder(with_rich=False, module_name="forge_builder_norich")
    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        core_folders=["docs"],
        base_files=["README.md"],
    )
    mod.create_structure(config)
    assert (tmp_path / "demo" / "docs").exists()


def test_create_structure_rich_errors(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_rich_error")

    def bad_mkdir(self, *args, **kwargs):
        if self.name == "boomdir":
            raise OSError("nope")
        return original_mkdir(self, *args, **kwargs)

    original_mkdir = Path.mkdir
    monkeypatch.setattr(Path, "mkdir", bad_mkdir)

    config = mod.ForgeConfig(
        name="demo",
        mode=mod.OperationMode.AUTONOMOUS,
        output_dir=tmp_path,
        core_folders=["boomdir"],
        base_files=[],
    )
    mod.create_structure(config)


def test_interactive_setup(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_setup")

    answers = iter(["myproj", str(tmp_path), "6"])

    def fake_prompt(_prompt, default=None):
        return next(answers)

    def fake_confirm(prompt, default=False):
        if "Save" in prompt:
            return True
        if "parallel" in prompt.lower():
            return True
        if "skip" in prompt.lower():
            return True
        return default

    monkeypatch.setattr(mod.Prompt, "ask", staticmethod(fake_prompt))
    monkeypatch.setattr(mod.Confirm, "ask", staticmethod(fake_confirm))
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    config = mod.ForgeConfig(name="demo")
    updated = mod.interactive_setup(mod.Console(), config)
    assert updated.name == "myproj"
    assert updated.output_dir == tmp_path
    assert updated.parallel is True
    assert (tmp_path / ".eidosian" / "forge_config.json").exists()


def test_main_rich_and_non_rich(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_main_rich")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    captured = {}

    def fake_interactive_setup(console, config):
        config.name = "from_setup"
        config.output_dir = tmp_path
        return config

    def fake_create(config):
        captured["config"] = config

    monkeypatch.setattr(mod, "interactive_setup", fake_interactive_setup)
    monkeypatch.setattr(mod, "create_structure", fake_create)
    monkeypatch.setattr(sys, "argv", ["forge_builder"])
    mod.main()

    assert captured["config"].name == "from_setup"
    assert captured["config"].modules

    mod_nr = load_forge_builder(
        with_rich=False, module_name="forge_builder_main_norich"
    )
    monkeypatch.setattr(mod_nr.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(mod_nr, "create_structure", fake_create)
    monkeypatch.setattr(sys, "argv", ["forge_builder", "demo"])
    mod_nr.main()
    assert captured["config"].mode == mod_nr.OperationMode.AUTONOMOUS


def test_main_modes_and_config(monkeypatch, tmp_path):
    mod = load_forge_builder(module_name="forge_builder_main_modes")
    monkeypatch.setattr(mod.Path, "home", lambda: tmp_path)

    captured = {}

    def fake_create(config):
        captured["config"] = config

    monkeypatch.setattr(mod, "create_structure", fake_create)
    monkeypatch.setattr(
        mod, "load_config_file", lambda _path: {"core_folders": ["src"]}
    )

    monkeypatch.setattr(
        sys, "argv", ["forge_builder", "--mode", "dry-run", "--config", "cfg.json"]
    )
    mod.main()
    assert captured["config"].mode == mod.OperationMode.DRY_RUN
    assert captured["config"].core_folders == ["src"]

    monkeypatch.setattr(sys, "argv", ["forge_builder", "--mode", "autonomous"])
    mod.main()
    assert captured["config"].mode == mod.OperationMode.AUTONOMOUS


def test_main_guard_runs_version(monkeypatch):
    mod = load_forge_builder(module_name="forge_builder_main_guard")
    monkeypatch.setattr(sys, "argv", ["forge_builder", "--version"])
    try:
        import runpy

        runpy.run_path(str(FORGE_PATH), run_name="__main__")
    except SystemExit:
        pass
