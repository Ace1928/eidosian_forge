from paste.httpexceptions import HTTPException
from wsgilib import catch_errors
def basic_transaction(environ, start_response):
    conn = factory(environ)
    environ['paste.connection'] = conn
    should_commit = [500]

    def finalizer(exc_info=None):
        if exc_info:
            if isinstance(exc_info[1], HTTPException):
                should_commit.append(exc_info[1].code)
        if should_commit.pop() < 400:
            conn.commit()
        else:
            try:
                conn.rollback()
            except:
                return
        conn.close()

    def basictrans_start_response(status, headers, exc_info=None):
        should_commit.append(int(status.split(' ')[0]))
        return start_response(status, headers, exc_info)
    return catch_errors(application, environ, basictrans_start_response, finalizer, finalizer)