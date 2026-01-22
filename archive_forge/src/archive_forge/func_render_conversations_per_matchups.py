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
def render_conversations_per_matchups(self):
    """
        Render conversations with and without reasons included.
        """
    matchups = list(self.dataframe.matchup.unique())

    def _render_row(matchup: List[str], row: pd.Series, row_id: int) -> str:
        dialogues = {'winner_dialogue': '', 'loser_dialogue': ''}
        for d_key in dialogues:
            result = []
            for _, turn in enumerate(row[d_key]['dialogue']):
                speakername = turn['id']
                text = turn['text']
                is_bot = speakername != 'human_evaluator' and speakername != 'other_speaker' and (speakername in matchup)
                align = 'right' if is_bot else 'left'
                color = 'white' if is_bot else 'black'
                bgcolor = '#2391f7' if is_bot else '#e1e1e7'
                result.append(f'<div style="overflow: auto; padding: 1ex 0;"><div style="clear: both; float: {align}; color: {color}; background-color: {bgcolor}; padding: 0.5em 1em; border-radius: 1em; max-width: 80%"><p style="margin: 0">{speakername}: {text}</p></div></div>')
            dialogues[d_key] = '<div style="background-color: white; margin: 0em; padding: 0.5em; font-family: sans-serif; font-size: 9pt; width: 99%;">' + ''.join(result) + '</div>'
        if row['winner'] == row['speaker_model_mapping'][0]:
            speakers_footnote = '(Speaker_1[winner] = {}, Speaker_2 = {})'.format(row['speaker_model_mapping'][0], row['speaker_model_mapping'][1])
        else:
            speakers_footnote = '(Speaker_1 = {}, Speaker_2[winner] = {})'.format(row['speaker_model_mapping'][0], row['speaker_model_mapping'][1])
        checkbox_row = '<td><div><input type= "checkbox" id= "cherry" name= "cherry"><label for="cherry">Cherry</label></div><div><input type= "checkbox" id= "lemon" name= "lemon"><label for= "lemon">Lemon</label></div><div><input type= "checkbox" id= "neutral" name= "neutral"><label for= "neutral">Neutral</label></div></td>'
        dialogue_row = f'<td>{dialogues['winner_dialogue']}</td><td>{dialogues['loser_dialogue']}</td>'
        reason_row = f'<td>{row['reason']}\n{speakers_footnote}</td>'
        if self.annotate_convo:
            return f'<tr><td>Pair {str(row_id)}</td>{checkbox_row}{dialogue_row}{reason_row}</tr>'
        else:
            return f'<tr><td>Pair {str(row_id)}</td>{dialogue_row}{reason_row}</tr>'

    def _render_html(table: pd.DataFrame) -> HTML:
        result = '                <div id="toc_container">                <p class="toc_title">Model Pairs</p>                <ul class="toc_list">'
        for matchup in matchups:
            eval_question = table.loc[table['matchup'] == matchup, 'question'].iloc[0]
            result += f"<li><a href='#{matchup}''>{matchup + '__on__' + eval_question}</a></li>"
        result += '</ul></div>'
        for matchup in matchups:
            length = min(self.max_matchups_html, len(table[table['matchup'] == matchup]))
            eval_question = table.loc[table['matchup'] == matchup, 'question'].iloc[0]
            matchup_table = table[table['matchup'] == matchup][:length]
            table_rows = [_render_row(matchup.split('__vs__'), row, idx) for idx, (_, row) in enumerate(matchup_table.iterrows())]
            if self.annotate_convo:
                table_body = f"<table border=1 frame=void rules=rows cellpadding='20'><tr><th>Pair Id</th><th>Comments</th><th>Winner Conversation</th><th>Loser Conversation</th><th>Reason</th></tr>{''.join(table_rows)}</table>"
            else:
                table_body = f"<table border=1 frame=void rules=rows cellpadding='20'><tr><th>Pair Id</th><th>Winner Conversation</th><th>Loser Conversation</th><th>Reason</th></tr>{''.join(table_rows)}</table>"
            result += f"<h2 id='{matchup}'><li><a href='#toc_container'>{matchup + '__on__' + eval_question}</a></li></h2><body>{table_body}</body>"
        return HTML(result)
    table = self.dataframe
    self.rendered_without_reasons = _render_html(table)
    table = table[table['reason'] != '']
    self.rendered_with_reasons = _render_html(table)