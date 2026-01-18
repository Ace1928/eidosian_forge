# tests/conftest.py
"""
Pytest configuration file for figlet_forge tests.

Provides fixtures and configuration for testing across different environments.
"""

import os
import platform
import sys
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import pytest

# Ensure package can be imported
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data constants
TEST_TEXT = "Hello, Figlet!"
TEST_UNICODE_TEXT = "Hello 世界"
STANDARD_FONTS = ["standard", "slant", "small", "big", "mini"]
TEST_COLORS = ["RED", "GREEN", "BLUE", "YELLOW", "CYAN", "MAGENTA"]


@pytest.fixture
def test_text() -> str:
    """Provide standard test text."""
    return TEST_TEXT


@pytest.fixture
def test_unicode_text() -> str:
    """Provide Unicode test text."""
    return TEST_UNICODE_TEXT


@pytest.fixture
def standard_fonts() -> List[str]:
    """Provide a list of standard fonts that should be available."""
    return STANDARD_FONTS


@pytest.fixture
def temp_file_path(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Provide a temporary file path for testing file operations.

    Args:
        tmp_path: Pytest's built-in temporary directory fixture

    Yields:
        A Path object pointing to a temporary file
    """
    temp_file = tmp_path / "test_output.txt"
    yield temp_file
    # Cleanup if file exists
    if temp_file.exists():
        temp_file.unlink()


@pytest.fixture
def captured_output() -> Generator[Tuple[List[str], List[str]], None, None]:
    """
    Capture stdout and stderr for testing.

    Yields:
        Tuple containing lists of captured stdout and stderr lines
    """
    import io
    from contextlib import redirect_stderr, redirect_stdout

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        yield stdout_capture, stderr_capture


@pytest.fixture
def figlet_factory():
    """
    Provide a factory function to create Figlet instances with specified parameters.

    Returns:
        Function that creates Figlet instances
    """
    from figlet_forge import Figlet

    def _create_figlet(**kwargs):
        """Create a Figlet instance with specified parameters."""
        return Figlet(**kwargs)

    return _create_figlet


@pytest.fixture
def mock_env_vars(monkeypatch):
    """
    Set up mock environment variables.

    Args:
        monkeypatch: pytest's monkeypatch fixture

    Returns:
        Dictionary of environment variables
    """
    env_vars = {"COLUMNS": "120", "LINES": "40"}

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    return env_vars


@pytest.fixture
def sample_figlet_string():
    """Provide a sample FigletString for testing transformations."""
    from figlet_forge.core.figlet_string import FigletString

    ascii_art = r"""
  ___ ___ ___
 | _ \_ _/ __|
 |  _/| | (_ |
 |_| |___\___|
"""
    return FigletString(ascii_art.strip())


@pytest.fixture
def temp_env(monkeypatch) -> Generator[Dict[str, str], None, None]:
    """
    Fixture to provide a clean environment dictionary that will be restored after the test.

    Returns:
        Dictionary to modify the environment with
    """
    original_env = os.environ.copy()
    temp_env_dict = {}

    yield temp_env_dict

    # Apply the temporary environment
    for key, value in temp_env_dict.items():
        monkeypatch.setenv(key, value)

    # Restore original environment after test
    for key in [k for k in os.environ if k not in original_env]:
        monkeypatch.delenv(key)
    for key, value in original_env.items():
        monkeypatch.setenv(key, value)


@pytest.fixture
def temp_directory(tmp_path) -> Generator[Path, None, None]:
    """
    Fixture to provide a temporary directory for file operations.

    Returns:
        Temporary directory path
    """
    original_dir = os.getcwd()
    os.chdir(tmp_path)

    yield tmp_path

    os.chdir(original_dir)


@pytest.fixture
def color_supported_terminal(monkeypatch) -> None:
    """
    Configure environment for a color-supporting terminal.
    """
    if platform.system() == "Windows":
        monkeypatch.setenv("TERM_PROGRAM", "vscode")
        monkeypatch.setenv("COLORTERM", "truecolor")
    else:
        monkeypatch.setenv("TERM", "xterm-256color")
        monkeypatch.setenv("COLORTERM", "truecolor")


@pytest.fixture
def no_color_terminal(monkeypatch) -> None:
    """
    Configure environment for a terminal that doesn't support color.
    """
    monkeypatch.setenv("NO_COLOR", "1")
    if "COLORTERM" in os.environ:
        monkeypatch.delenv("COLORTERM")
    if platform.system() != "Windows":
        monkeypatch.setenv("TERM", "dumb")


@pytest.fixture
def unicode_supported_terminal(monkeypatch) -> None:
    """
    Configure environment for a Unicode-supporting terminal.
    """
    monkeypatch.setenv("LANG", "en_US.UTF-8")
    if platform.system() == "Windows":
        monkeypatch.setenv("TERM_PROGRAM", "vscode")
    else:
        monkeypatch.setenv("LC_ALL", "en_US.UTF-8")


@pytest.fixture
def ascii_only_terminal(monkeypatch) -> None:
    """
    Configure environment for an ASCII-only terminal.
    """
    monkeypatch.setenv("LANG", "C")
    if platform.system() == "Windows":
        # Windows rarely has true ASCII-only terminals now, but we'll try
        if "TERM_PROGRAM" in os.environ:
            monkeypatch.delenv("TERM_PROGRAM")
    else:
        monkeypatch.setenv("LC_ALL", "C")


# Register markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "windows_only: mark test to run only on Windows")
    config.addinivalue_line(
        "markers", "unix_only: mark test to run only on Unix-like systems"
    )
    config.addinivalue_line("markers", "color: mark test that requires color support")
    config.addinivalue_line(
        "markers", "unicode: mark test that requires Unicode support"
    )


# Skip tests based on platform
def pytest_runtest_setup(item):
    """Set up test runs with conditional skipping."""
    for marker in item.iter_markers(name="windows_only"):
        if platform.system() != "Windows":
            pytest.skip("Windows-only test")

    for marker in item.iter_markers(name="unix_only"):
        if platform.system() == "Windows":
            pytest.skip("Unix-only test")
