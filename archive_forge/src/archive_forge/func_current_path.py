from keras.src.backend.common import global_state
def current_path():
    name_scope_stack = global_state.get_global_attribute('name_scope_stack')
    if name_scope_stack is None:
        return ''
    return '/'.join((x.name for x in name_scope_stack))