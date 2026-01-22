import torch
def m_fn_rec(m, d):
    new_m = m_fn(m, d)
    for name, sub_m in m.named_children():
        setattr(new_m, name, m_fn_rec(sub_m, d))
    return new_m