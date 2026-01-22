import greenlet
def c_run():
    results.append('Begin C')
    b.switch('From C')
    results.append('C done')