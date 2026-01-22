from concurrent.futures import ThreadPoolExecutor
from promise import Promise
import time
import weakref
import gc
def test_issue_26():
    context = {'success': False}
    promise1 = Promise(lambda resolve, reject: context.update({'promise1_reject': reject}))
    promise1.then(lambda x: None)
    promise1.then(lambda x: None)
    context['promise1_reject'](RuntimeError('Ooops!'))
    promise2 = Promise(lambda resolve, reject: context.update({'promise2_resolve': resolve}))
    promise3 = promise2.then(lambda x: context.update({'success': True}))
    context['promise2_resolve'](None)
    promise3._wait(timeout=0.1)
    assert context['success']