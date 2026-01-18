"""Entry point for running the Eidosian Universe simulation."""

from __future__ import annotations

import sys
import pygame

# When executed directly, ``__package__`` will be ``None`` and relative imports
# fail.  To support ``python eu_main.py`` from within the ``src`` directory we
# manually adjust ``sys.path`` and use absolute imports.  When run as a module
# (``python -m eidosian_universe.eu_main``) the relative imports work normally.

if __package__ in (None, ""):
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from eidosian_universe.eu_config import UniverseConfig
    from eidosian_universe.eu_world import Universe
    from eidosian_universe.eu_ai import EidosAI
else:
    from .eu_config import UniverseConfig
    from .eu_world import Universe
    from .eu_ai import EidosAI


def main() -> None:
    """Run the Eidosian Universe simulation."""
    pygame.init()
    config = UniverseConfig()
    flags = pygame.FULLSCREEN if config.fullscreen else 0
    screen = pygame.display.set_mode((config.width, config.height), flags)
    clock = pygame.time.Clock()
    universe = Universe(config)
    ai = EidosAI(universe)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                universe.agents.append(universe._create_random_agent(x=x, y=y))
            if event.type == pygame.FINGERDOWN:
                x = int(event.x * config.width)
                y = int(event.y * config.height)
                universe.agents.append(universe._create_random_agent(x=x, y=y))

        ai.update()
        universe.update()
        universe.render(screen)
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
