import pytest
from string import ascii_letters
from random import randint
import gc
import sys
@pytest.mark.parametrize('n', [1, 10, 100, 1000])
@pytest.mark.parametrize('name', ['label', 'button'])
@pytest.mark.parametrize('tick', ['tick', 'no_tick'])
def test_random_label_create(kivy_benchmark, n, name, tick):
    from kivy.clock import Clock
    from kivy.uix.label import Label
    from kivy.uix.button import Button
    label = Label(text='*&^%')
    button = Button(text='*&^%')
    cls = Label if name == 'label' else Button
    labels = []
    k = len(ascii_letters)
    for x in range(n):
        label = [ascii_letters[randint(0, k - 1)] for _ in range(10)]
        labels.append(''.join(label))

    def make_labels():
        o = []
        for text in labels:
            o.append(cls(text=text))
        if tick == 'tick':
            Clock.tick()
    kivy_benchmark(make_labels)