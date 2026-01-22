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
def create_agent_by_name(self, name, device, maxlen):
    agent_name = name.split('.')[0]
    if agent_name == 'blender':
        return BlenderGenerationAgent(name, device, maxlen)
    elif agent_name == 'gptneo':
        return GPTNeoAgent(name, device, maxlen)
    elif agent_name == 'dialogpt':
        return DialoGPTAgent(name, device, maxlen)
    elif agent_name == 'dodecathlon':
        return DodecathlonAgent(name, device, maxlen)
    elif agent_name == 'reddit':
        return RedditAgent(name, device, maxlen)
    elif agent_name == 'unlikelihood':
        return UnlikelihoodAgent(name, device, maxlen)
    elif agent_name == 'wizard_of_wikipedia':
        return WizardOfWikipediaGenerationAgent(name, device, maxlen)
    elif agent_name == 'safety':
        if name.split('.')[1] == 'offensive':
            return OffensiveAgent(name, device, maxlen)
        elif name.split('.')[1] == 'sensitive':
            return SensitiveAgent(name, device, maxlen)
        else:
            return Exception('wrong model')
    else:
        return Exception('wrong model')