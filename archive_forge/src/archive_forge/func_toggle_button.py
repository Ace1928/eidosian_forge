import logging
import sys
import warnings
from typing import Optional
import wandb
def toggle_button(what='run'):
    return f"""<button onClick="this.nextSibling.style.display='block';this.style.display='none';">Display W&B {what}</button>"""