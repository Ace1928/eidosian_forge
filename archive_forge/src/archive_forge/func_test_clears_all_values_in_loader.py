from pytest import raises
from promise import Promise, async_instance
from promise.dataloader import DataLoader
def test_clears_all_values_in_loader():

    @Promise.safe
    def do():
        identity_loader, load_calls = id_loader()
        a, b = Promise.all([identity_loader.load('A'), identity_loader.load('B')]).get()
        assert a == 'A'
        assert b == 'B'
        assert load_calls == [['A', 'B']]
        identity_loader.clear_all()
        a2, b2 = Promise.all([identity_loader.load('A'), identity_loader.load('B')]).get()
        assert a2 == 'A'
        assert b2 == 'B'
        assert load_calls == [['A', 'B'], ['A', 'B']]
    do().get()