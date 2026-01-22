from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent
from parlai.core.message import Message
from parlai.core.params import ParlaiParser
from parlai.utils.misc import TimeLogger, warn_once
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
def verify_data(opt, parser):
    report_text, report_log = verify(parser.parse_args())
    print(report_text)