import urllib
from typing import Optional
from wandb_gql import gql
import wandb
from wandb import util
from wandb.apis import public
from wandb.apis.attrs import Attrs
from wandb.sdk.lib import ipython
def best_run(self, order=None):
    """Return the best run sorted by the metric defined in config or the order passed in."""
    if order is None:
        order = self.order
    else:
        order = public.QueryGenerator.format_order_key(order)
    if order is None:
        wandb.termwarn("No order specified and couldn't find metric in sweep config, returning most recent run")
    else:
        wandb.termlog('Sorting runs by %s' % order)
    filters = {'$and': [{'sweep': self.id}]}
    try:
        return public.Runs(self.client, self.entity, self.project, order=order, filters=filters, per_page=1)[0]
    except IndexError:
        return None