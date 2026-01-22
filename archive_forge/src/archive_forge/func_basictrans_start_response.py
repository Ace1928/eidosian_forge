from paste.httpexceptions import HTTPException
from wsgilib import catch_errors
def basictrans_start_response(status, headers, exc_info=None):
    should_commit.append(int(status.split(' ')[0]))
    return start_response(status, headers, exc_info)