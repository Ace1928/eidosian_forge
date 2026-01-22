from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.clock import Clock
import timeit
def start_test(self, *largs):
    self.but.text = 'test started'
    self.slider.max = len(self.tests)
    ev = None

    def test(*l):
        if self.test_done:
            try:
                but = self.tests[int(self.slider.value)]
                self.slider.value += 1
                but.state = 'down'
                print('=====================')
                print('Test:', but.text)
                print('=====================')
                but.test(but)
            except IndexError:
                for but in self.tests:
                    but.state = 'normal'
                self.but.text = 'Start Test'
                self.slider.value = 0
                print('===================')
                print('All Tests Completed')
                print('===================')
                ev.cancel()
    ev = Clock.schedule_interval(test, 1)