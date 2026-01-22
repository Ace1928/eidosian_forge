import linecache
import re
def clear_trace_filter_cache():
    """
    Clear the trace filter cache.
    Call this after reloading.
    """
    global should_trace_hook
    try:
        old_hook = should_trace_hook
        should_trace_hook = None
        linecache.clearcache()
        _filename_to_ignored_lines.clear()
    finally:
        should_trace_hook = old_hook