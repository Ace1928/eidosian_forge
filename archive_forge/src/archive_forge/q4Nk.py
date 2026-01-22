import pyautogui
import time

# Wait for 2 seconds to allow the user to switch to the desired window
time.sleep(2)
# Open the text editor (assuming it's in the Applications folder on macOS)
pyautogui.press("command")
pyautogui.typewrite("space")
pyautogui.typewrite("textedit")
pyautogui.press("enter")
# Wait for the text editor to open
time.sleep(2)
# Type the message
pyautogui.typewrite("Hello, World!")
# Save the file
pyautogui.hotkey("command", "s")
pyautogui.typewrite("message.txt")
pyautogui.press("enter")
# Close the text editor
pyautogui.hotkey("command", "q")
