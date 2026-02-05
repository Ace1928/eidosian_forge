import subprocess
import sys


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
