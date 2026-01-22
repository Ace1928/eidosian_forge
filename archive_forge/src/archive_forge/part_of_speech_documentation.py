import wandb
from wandb import util
from wandb.plots.utils import (

    Adds support for spaCy's dependency visualizer which shows
        part-of-speech tags and syntactic dependencies.

    Arguments:
     docs (list, Doc, Span): Document(s) to visualize.

    Returns:
     Nothing. To see plots, go to your W&B run page.

    Example:
     wandb.log({'part_of_speech': wandb.plots.POS(docs=doc)})
    