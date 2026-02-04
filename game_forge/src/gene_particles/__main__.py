"""Module entrypoint for Gene Particles."""

from game_forge.src.gene_particles.gp_main import _entrypoint


def run() -> int:
    """Run the Gene Particles CLI entrypoint."""
    return _entrypoint()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(run())
