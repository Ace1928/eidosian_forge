import statemachine
def press_alpha_button(self):
    try:
        super(VendingMachine, self).press_alpha_button()
    except VendingMachineState.InvalidTransitionException as ite:
        print(ite)
    else:
        self._alpha_pressed = self._pressed