from parlai.core.params import ParlaiParser
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.worlds import create_task
from parlai.utils.misc import TimeLogger
from parlai.core.dict import DictionaryAgent
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
def keep_token(t):
    for s in ignore_tokens:
        if s != '' and s in t:
            return False
    return True