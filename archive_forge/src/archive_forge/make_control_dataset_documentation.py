from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import msg_to_str, TimeLogger
from controllable_seq2seq.util import ConvAI2History
from controllable_seq2seq.controls import eval_attr, initialize_control_information
import random

Make a copy of the ConvAI2 dataset with CT control variables annotated.
