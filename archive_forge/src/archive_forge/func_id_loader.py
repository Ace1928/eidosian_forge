from pytest import mark
from promise import Promise
from promise.dataloader import DataLoader
def id_loader(**options):
    load_calls = []
    resolve = options.pop('resolve', Promise.resolve)

    def fn(keys):
        load_calls.append(keys)
        return resolve(keys)
    identity_loader = DataLoader(fn, **options)
    return (identity_loader, load_calls)