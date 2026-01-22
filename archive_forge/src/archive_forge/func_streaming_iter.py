import paste.util.threadinglocal as threadinglocal
def streaming_iter(self, reg, environ, start_response):
    try:
        for item in self.application(environ, start_response):
            yield item
    except Exception as e:
        if environ.get('paste.evalexception'):
            expected = False
            for expect in environ.get('paste.expected_exceptions', []):
                if isinstance(e, expect):
                    expected = True
            if not expected:
                restorer.save_registry_state(environ)
        reg.cleanup()
        raise
    except:
        if environ.get('paste.evalexception'):
            restorer.save_registry_state(environ)
        reg.cleanup()
        raise
    else:
        reg.cleanup()