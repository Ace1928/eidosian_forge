import re
def task_lists_hook(md, state):
    return _rewrite_all_list_items(state.tokens)