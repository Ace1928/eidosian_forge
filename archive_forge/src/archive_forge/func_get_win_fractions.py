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
def get_win_fractions(self) -> pd.DataFrame:
    """
        Return the joined matchup + win totals, get win fractions.

        Sorted according to win percentage
        """
    if not hasattr(self, 'win_total_df'):
        self.get_wins_per_model_matchup()
    self.win_fraction_df = self.matchup_total_df.join(self.win_total_df).assign(win_frac=lambda df: df['win_total'] / df['matchup_total'])
    pivoted_df = self.win_fraction_df.pivot(index='loser', columns='winner', values='win_frac')
    self.models_by_win_frac = self.win_fraction_df.groupby('winner')['win_frac'].mean().sort_values().index.values.tolist()
    self.sorted_win_frac_df = pivoted_df.reindex(index=self.models_by_win_frac, columns=self.models_by_win_frac)
    return self.sorted_win_frac_df