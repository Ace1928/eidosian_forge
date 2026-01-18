import unittest
import importlib.util
import importlib.machinery
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMART_PUBLISH_PATH = ROOT / "smart_publish"
PY_LIB_PATH = ROOT / "lib" / "py_lib.py"


def load_module(name: str, path: Path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    spec = importlib.util.spec_from_loader(name, loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


smart_publish = load_module("smart_publish", SMART_PUBLISH_PATH)
py_lib = load_module("py_lib", PY_LIB_PATH)


class TestUtilities(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(smart_publish.parse_version("1.2.3"), (1, 2, 3))

    def test_increment_version_patch(self):
        self.assertEqual(
            smart_publish.increment_version("1.2.3", smart_publish.VersionBump.PATCH),
            "1.2.4",
        )

    def test_increment_version_minor(self):
        self.assertEqual(
            smart_publish.increment_version("1.2.3", smart_publish.VersionBump.MINOR),
            "1.3.0",
        )

    def test_validate_version(self):
        smart_publish.validate_version("1.2.3")
        with self.assertRaises(ValueError):
            smart_publish.validate_version("bad.version")

    def test_normalize_exit_code(self):
        self.assertEqual(py_lib.normalize_exit_code(0), 0)
        self.assertEqual(py_lib.normalize_exit_code(2), 1)


if __name__ == "__main__":
    unittest.main()
