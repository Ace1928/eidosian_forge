import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy.stats import binom_test
from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML
def filter_by_dialogue_length(self, is_debug=False):
    """
        Filter out matchup with one of the conversation shorter than
        self.min_dialogue_length This applies to calculating sorted_win_frac_df and
        signficance_df, but not html visualizations of conversations.

        :param is_debug: if True, print logs indicating the number of pairings filtered out due to short conversation.
            is_debug bool
        """
    df = pd.DataFrame()
    filter_list = {}
    for _, row in self.dataframe.iterrows():
        keep_row = True
        for model_name, dialogue_length in row['dialogue_lengths'].items():
            if keep_row and dialogue_length < self.min_dialogue_length:
                keep_row = False
                filter_list[model_name] = filter_list.get(model_name, 0) + 1
        if keep_row:
            df = df.append(row, ignore_index=True)
    if is_debug:
        for model_name in filter_list:
            print(f'For {self.run_id}: filter out {filter_list[model_name]} matchups due to {model_name} with dialogue length shorter than {self.min_dialogue_length}')
    return df