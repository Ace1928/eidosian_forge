import subprocess
import sys


def test_style():
    assert (
        subprocess.run(
            [
                sys.executable,
                "-m",
                "black",
                "--check",
                "--diff",
                "core",
                "agents",
                "labs",
                "tools",
                "tests",
            ],
            check=False,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            [
                sys.executable,
                "-m",
                "flake8",
                "core",
                "agents",
                "labs",
                "tools",
                "tests",
            ],
            check=False,
        ).returncode
        == 0
    )
