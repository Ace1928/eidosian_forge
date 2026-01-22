from minerl.herobraine.hero.handlers.agent.action import Action
import jinja2
import minerl.herobraine.hero.spaces as spaces
import numpy as np
def xml_template(self) -> str:
    return str('<CameraCommands/>')