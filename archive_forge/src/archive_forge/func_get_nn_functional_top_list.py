def get_nn_functional_top_list():
    top_nn_functional_ = dict(top_nn_functional)
    for _, count, functional_name in top_nn_module:
        if functional_name is None:
            continue
        if functional_name == 'torch.flatten':
            continue
        if functional_name not in top_nn_functional_:
            top_nn_functional_[functional_name] = count
        else:
            top_nn_functional_[functional_name] += count
    top_nn_functional_ = list(top_nn_functional_.items())
    top_nn_functional_.sort(key=lambda x: x[1], reverse=True)
    return top_nn_functional_