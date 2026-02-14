LLM to LLM Chat

This program aims to use pyautogui to watch a pair of chat windows and send messages between the two bots as they finish, effectively facilitating a conversation between the two bots. This is a proof of concept for a larger project that will use this functionality to facilitate a conversation between two bots in a language learning context.
It will do so by:
1. Watching the chat windows for new messages
2. Sending the message to the other chat window
3. Waiting for the other chat window to respond
4. Sending the response to the original chat window
5. Repeat until the conversation is over
Saving a copy of the conversation to a file as it goes.

## Getting Started
1. Install the required packages
```bash
pip install pyautogui
```
2. Run the program
```bash
python llm_chat.py
```

''' python
import pyautogui
import time
import random

# Set the chat window locations
chat1 = (100, 100)
chat2 = (200, 200)

# Set the delay between messages
delay = 1

# Set the conversation length
conversation_length = 0 # 0 for infinite conversation

# Set the conversation file
conversation_file = 'conversation.txt'

# Start the conversation
conversation = []

while conversation_length == 0 or len(conversation) < conversation_length:
    # Watch for new messages in chat1
    pyautogui.moveTo(chat1)
    pyautogui.click()
    time.sleep(0.5)
    message = pyautogui.screenshot(region=(chat1[0], chat1[1], 200, 20))
    message.save('message.png')

    # Send the message to chat2
    pyautogui.moveTo(chat2)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.typewrite('Hello from chat1!')
    pyautogui.press('enter')
    conversation.append('Chat1: Hello from chat1!')

    # Watch for new messages in chat2
    pyautogui.moveTo(chat2)
    pyautogui.click()
    time.sleep(0.5)
    message = pyautogui.screenshot(region=(chat2[0], chat2[1], 200, 20))
    message.save('message.png')

    # Send the message to chat1
    pyautogui.moveTo(chat1)
    pyautogui.click()
    time.sleep(0.5)
    pyautogui.typewrite('Hello from chat2!')
    pyautogui.press('enter')
    conversation.append('Chat2: Hello from chat2!')

    # Save the conversation to a file
    with open(conversation_file, 'w') as file:
        for line in conversation:
            file.write(line + '\n')

    # Wait for the delay
    time.sleep(delay)

# End the conversation
print('Conversation ended.')
'''

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
'''

'''
## Acknowledgments
- [PyAutoGUI](https://pyautogui.readthedocs.io/en/latest/) - A Python module for programmatically controlling the mouse and keyboard.
'''
'''
## Authors
- **[Lloyd Handyside](https://github.com/LloydHandyside)** - *Initial work* - [Lloyd Handyside](https://github.com/LloydHandyside)

See also the list of [contributors](https://github.com/LloydHandyside/llm_chat/contributors) who participated in this project.


'''

'''
## Contributing
Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.
'''

'''
## Versioning
We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](
'''

'''
## Roadmap
See the [open issues](https://github.com/LloydHandyside/llm_chat/issues) for a list of proposed features (and known issues).


'''

'''
## Code of Conduct
Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) for details on our code of conduct.


'''

'''
## Changelog
See the [CHANGELOG.md](CHANGELOG.md) for details on the changes made in each version.


'''

'''
## Support
Please read [SUPPORT.md](SUPPORT.md) for details on our support policy.


'''