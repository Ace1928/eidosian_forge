from game_forge.src.gene_particles import gp_main
from game_forge.src.gene_particles import __main__ as gp_cli


def test_main_no_loop():
    result = gp_main.main(run_loop=False)
    assert result is None


def test_main_module_entrypoint(monkeypatch):
    monkeypatch.setenv("GENE_PARTICLES_TEST_MODE", "1")
    assert gp_main.run() == 0


def test_package_module_entrypoint(monkeypatch):
    monkeypatch.setenv("GENE_PARTICLES_TEST_MODE", "1")
    assert gp_cli.run() == 0


def test_entrypoint_modes(monkeypatch):
    monkeypatch.setenv("GENE_PARTICLES_TEST_MODE", "1")
    assert gp_main._entrypoint() == 0

    def fake_loop(self):
        return None

    monkeypatch.delenv("GENE_PARTICLES_TEST_MODE", raising=False)
    monkeypatch.setattr(gp_main.CellularAutomata, "main_loop", fake_loop)
    assert gp_main._entrypoint() == 0
