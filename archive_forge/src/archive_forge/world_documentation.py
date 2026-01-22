import jinja2
from minerl.herobraine.hero.handler import Handler
Generates a world using minecraft procedural generation.

        Args:
            force_reset (bool, optional): If the world should be reset every episode.. Defaults to True.
            generator_options: A JSON object specifying parameters to the procedural generator.
        