import os
def should_run_eventlet_tests():
    return bool(int(os.environ.get('TEST_EVENTLET') or '0'))