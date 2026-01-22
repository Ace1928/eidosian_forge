import random
from typing import Optional
from parlai.core.message import Message
from parlai.core.opt import Opt
def get_temp_history(self, observation: Message) -> Optional[str]:
    """
        Conditionally return a style-token string to temporarily insert into history.
        """
    use_style_rand = random.random()
    if use_style_rand < self.use_style_frac:
        style = observation.get('personality')
    else:
        style = ''
    if style is not None and style != '':
        return STYLE_SEP_TOKEN + style