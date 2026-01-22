from parlai.agents.transformer.transformer import TransformerClassifierAgent
from parlai.core.build_data import modelzoo_path
from parlai.utils.safety import OffensiveStringMatcher
from parlai.core.agents import add_datapath_and_model_args, create_agent_from_opt_file, create_agent
from openchat.base import ParlaiClassificationAgent, EncoderLM, SingleTurn

        Returns the probability that a message is safe according to the classifier.
        