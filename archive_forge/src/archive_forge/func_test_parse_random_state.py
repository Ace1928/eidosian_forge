import numpy as np
import cirq
def test_parse_random_state():
    global_state = np.random.get_state()

    def rand(prng):
        np.random.set_state(global_state)
        return prng.rand()
    prngs = [np.random, cirq.value.parse_random_state(np.random), cirq.value.parse_random_state(None)]
    vals = [rand(prng) for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)
    seed = np.random.randint(2 ** 31)
    prngs = [np.random.RandomState(seed), cirq.value.parse_random_state(np.random.RandomState(seed)), cirq.value.parse_random_state(seed)]
    vals = [prng.rand() for prng in prngs]
    eq = cirq.testing.EqualsTester()
    eq.add_equality_group(*vals)