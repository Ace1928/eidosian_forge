import datetime
import time
from .search import parse_search_terms, satisfies_search_terms
def get_task_by_id(events, task_id):
    return events.state.tasks.get(task_id)