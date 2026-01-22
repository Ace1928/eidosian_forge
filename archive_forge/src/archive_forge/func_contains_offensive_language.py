from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file, create_agent
from openchat.base import ParlaiClassificationAgent, EncoderLM, SingleTurn
def contains_offensive_language(self, text):
    """
        Returns the probability that a message is safe according to the classifier.
        """
    act = {'text': text, 'episode_done': True}
    self.agent.observe(act)
    response = self.agent.act()['text']
    pred_class, prob = [x.split(': ')[-1] for x in response.split('\n')]
    pred_not_ok = self.labels()[0 if pred_class == '__ok__' else 1]
    return pred_not_ok