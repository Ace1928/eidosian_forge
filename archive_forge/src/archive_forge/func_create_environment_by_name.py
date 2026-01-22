from openchat.agents.blender import BlenderGenerationAgent
from openchat.agents.dialogpt import DialoGPTAgent
from openchat.agents.dodecathlon import DodecathlonAgent
from openchat.agents.gptneo import GPTNeoAgent
from openchat.agents.safety import OffensiveAgent, SensitiveAgent
from openchat.agents.reddit import RedditAgent
from openchat.agents.unlikelihood import UnlikelihoodAgent
from openchat.agents.wow import WizardOfWikipediaGenerationAgent
from openchat.envs.interactive import InteractiveEnvironment
from openchat.utils.terminal_utils import draw_openchat
def create_environment_by_name(self, name):
    if name == 'interactive':
        return InteractiveEnvironment()
    elif name == 'interactive_web':
        return InteractiveWebEnvironment()
    elif name == 'webserver':
        raise NotImplemented
    elif name == 'facebook':
        raise NotImplemented
    elif name == 'kakaotalk':
        raise NotImplemented
    elif name == 'whatsapp':
        raise NotImplemented