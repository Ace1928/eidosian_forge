def store_c_uuid():
    global c_uuid
    c_uuid = next(iter(reality.resources_by_logical_name('C'))).uuid