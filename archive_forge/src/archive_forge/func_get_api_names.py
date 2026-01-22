import threading
import rtmidi
from .. import ports
from ..messages import Message
from ._parser_queue import ParserQueue
from .rtmidi_utils import expand_alsa_port_name
def get_api_names():
    return [_api_to_name[n] for n in rtmidi.get_compiled_api()]