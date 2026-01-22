import re
import warnings
from typing import Optional
import torch
from accelerate.utils import extract_model_from_parallel
from transformers import StoppingCriteria, StoppingCriteriaList
from ..import_utils import is_rich_available
def show_text(self, show_legend=False):
    """
        Print the text history.
        """
    if not is_rich_available():
        warnings.warn('install rich to display text')
        return
    text = Text(self.text)
    text.stylize(self.prompt_color, self.text_spans[0][0], self.text_spans[1][0])
    for i, (start, end) in enumerate(self.text_spans[1:]):
        if self.system_spans[i + 1]:
            text.stylize(self.system_color, start, end)
        else:
            text.stylize(self.model_color, start, end)
    text.append(f'\n\nReward: {self.reward}', style=self.reward_color)
    print(text)
    if show_legend:
        self.show_colour_legend()