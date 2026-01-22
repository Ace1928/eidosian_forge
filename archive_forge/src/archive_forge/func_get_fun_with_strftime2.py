import dill
def get_fun_with_strftime2():
    import datetime
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')