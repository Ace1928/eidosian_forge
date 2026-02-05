import subprocess
import sys
from pathlib import Path


def test_gene_particles_benchmark_help():
    result = subprocess.run(
        [sys.executable, "game_forge/tools/gene_particles_benchmark.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Benchmark Gene Particles" in result.stdout


def test_gene_particles_profile_help():
    result = subprocess.run(
        [sys.executable, "game_forge/tools/gene_particles_profile.py", "--help"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Profile Gene Particles" in result.stdout


def test_gene_particles_profile_output(tmp_path: Path) -> None:
    output = tmp_path / "profile.stats"
    result = subprocess.run(
        [
            sys.executable,
            "game_forge/tools/gene_particles_profile.py",
            "--steps",
            "1",
            "--top",
            "1",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "INFO wrote profile" in result.stdout
    assert output.exists()
