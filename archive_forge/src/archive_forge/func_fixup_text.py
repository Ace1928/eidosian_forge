from typing import TYPE_CHECKING, List, Tuple, cast, Dict
import matplotlib.textpath
import matplotlib.font_manager
def fixup_text(text: str):
    if '\n' in text:
        return '?'
    text = text.replace('[<virtual>]', '')
    text = text.replace('[cirq.VirtualTag()]', '')
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    return text