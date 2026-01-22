import random
def run_debate(self):
    print(f'Discussing: {self.current_topic}')
    for _ in range(3):
        user_input = input('Enter your argument: ')
        print('Analyzing argument...')
        if 'rights' in user_input.lower():
            print('Interesting point on rights. How do you think this aligns with human rights?')
        else:
            print("Thank you for your input. Let's consider other aspects as well.")