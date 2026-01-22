import sys
import re
import joblib
def write_test_pickle(to_pickle, args):
    kwargs = {}
    compress = args.compress
    method = args.method
    joblib_version = get_joblib_version()
    py_version = '{0[0]}{0[1]}'.format(sys.version_info)
    numpy_version = ''.join(np.__version__.split('.')[:2])
    body = '_compressed' if compress and method == 'zlib' else ''
    if compress:
        if method == 'zlib':
            kwargs['compress'] = True
            extension = '.gz'
        else:
            kwargs['compress'] = (method, 3)
            extension = '.pkl.{}'.format(method)
        if args.cache_size:
            kwargs['cache_size'] = 0
            body += '_cache_size'
    else:
        extension = '.pkl'
    pickle_filename = 'joblib_{}{}_pickle_py{}_np{}{}'.format(joblib_version, body, py_version, numpy_version, extension)
    try:
        joblib.dump(to_pickle, pickle_filename, **kwargs)
    except Exception as e:
        print("Error: cannot generate file '{}' with arguments '{}'. Error was: {}".format(pickle_filename, kwargs, e))
    else:
        print("File '{}' generated successfully.".format(pickle_filename))