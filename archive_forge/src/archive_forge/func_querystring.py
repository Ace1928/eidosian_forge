def querystring(*args, **kwargs):
    return intersect(*args, **kwargs).to_string()