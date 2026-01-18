import subprocess


def test_style():
    assert (
        subprocess.run(
            ["black", "--check", "--diff", "core", "agents", "labs", "tools", "tests"],
            check=False,
        ).returncode
        == 0
    )
    assert (
        subprocess.run(
            ["flake8", "core", "agents", "labs", "tools", "tests"], check=False
        ).returncode
        == 0
    )
