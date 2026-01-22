from parlai.core.message import Message
from parlai.utils.misc import display_messages
from parlai.utils.strings import colorize
from parlai.agents.local_human.local_human import LocalHumanAgent
from parlai.utils.safety import OffensiveStringMatcher, OffensiveLanguageClassifier
def check_offensive(self, text):
    """
        Check if text is offensive using string matcher and classifier.
        """
    if text == '':
        return False
    if hasattr(self, 'offensive_string_matcher') and text in self.offensive_string_matcher:
        return True
    if hasattr(self, 'offensive_classifier') and text in self.offensive_classifier:
        return True
    return False