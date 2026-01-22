import greenlet
def g1_run():
    print('In g1_run')
    global switch_to_g2
    switch_to_g2 = True
    greenlet.getcurrent().parent.switch()
    print('Return to g1_run')
    print('Falling off end of g1_run')