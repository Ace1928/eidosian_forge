from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.agents.repeat_query.repeat_query import RepeatQueryAgent
import parlai.utils.logging as logging
import random
import cProfile
import io
import pstats

Basic script which allows to profile interaction with a model using repeat_query to
avoid human interaction (so we can time it, only).
