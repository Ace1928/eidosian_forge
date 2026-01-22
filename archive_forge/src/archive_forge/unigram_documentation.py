import json
import re
from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from itertools import islice

        Stub load which ignores the model on disk, as UnigramAgent depends on the
        dictionary, which is saved elsewhere.
        