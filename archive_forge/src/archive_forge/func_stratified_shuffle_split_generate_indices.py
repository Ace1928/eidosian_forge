import numpy as np
def stratified_shuffle_split_generate_indices(y, n_train, n_test, rng, n_splits=10):
    """

    Provides train/test indices to split data in train/test sets.
    It's reference is taken from StratifiedShuffleSplit implementation
    of scikit-learn library.

    Args
    ----------

    n_train : int,
        represents the absolute number of train samples.

    n_test : int,
        represents the absolute number of test samples.

    random_state : int or RandomState instance, default=None
        Controls the randomness of the training and testing indices produced.
        Pass an int for reproducible output across multiple function calls.

    n_splits : int, default=10
        Number of re-shuffling & splitting iterations.
    """
    classes, y_indices = np.unique(y, return_inverse=True)
    n_classes = classes.shape[0]
    class_counts = np.bincount(y_indices)
    if np.min(class_counts) < 2:
        raise ValueError('Minimum class count error')
    if n_train < n_classes:
        raise ValueError('The train_size = %d should be greater or equal to the number of classes = %d' % (n_train, n_classes))
    if n_test < n_classes:
        raise ValueError('The test_size = %d should be greater or equal to the number of classes = %d' % (n_test, n_classes))
    class_indices = np.split(np.argsort(y_indices, kind='mergesort'), np.cumsum(class_counts)[:-1])
    for _ in range(n_splits):
        n_i = approximate_mode(class_counts, n_train, rng)
        class_counts_remaining = class_counts - n_i
        t_i = approximate_mode(class_counts_remaining, n_test, rng)
        train = []
        test = []
        for i in range(n_classes):
            permutation = rng.permutation(class_counts[i])
            perm_indices_class_i = class_indices[i].take(permutation, mode='clip')
            train.extend(perm_indices_class_i[:n_i[i]])
            test.extend(perm_indices_class_i[n_i[i]:n_i[i] + t_i[i]])
        train = rng.permutation(train)
        test = rng.permutation(test)
        yield (train, test)