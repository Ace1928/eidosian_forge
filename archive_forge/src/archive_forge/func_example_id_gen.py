def example_id_gen(max_n=1000):
    for i in range(1, max_n):
        yield ('pythree_example_model_%03d' % (i,))