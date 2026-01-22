import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtWidgets import (
    QMainWindow,
    QVBoxLayout,
    QWidget,
    QLineEdit,
    QPushButton,
    QTextEdit,
)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
import pyautogui
import threading
import time


class BrowserBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.driver = None

    def init_ui(self):
        self.setWindowTitle("ChatBot Controlled Browser")
        self.setGeometry(100, 100, 1200, 800)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.browser = QWebEngineView()
        self.layout.addWidget(self.browser)

        self.url_bar = QLineEdit()
        self.url_bar.setPlaceholderText("Enter URL and press Enter")
        self.url_bar.returnPressed.connect(self.load_url)
        self.layout.addWidget(self.url_bar)

        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Enter command and press Enter")
        self.chat_input.returnPressed.connect(self.process_command)
        self.layout.addWidget(self.chat_input)

        self.chat_output = QTextEdit()
        self.chat_output.setReadOnly(True)
        self.layout.addWidget(self.chat_output)

        self.start_selenium()

    def start_selenium(self):
        self.driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install())
        )

    def load_url(self):
        url = self.url_bar.text()
        self.browser.setUrl(QtCore.QUrl(url))
        if self.driver:
            self.driver.get(url)

    def process_command(self):
        command = self.chat_input.text()
        self.chat_output.append(f"Command: {command}")
        self.chat_input.clear()

        if command.startswith("goto "):
            url = command.split(" ", 1)[1]
            self.url_bar.setText(url)
            self.load_url()
        elif command.startswith("click "):
            element = command.split(" ", 1)[1]
            self.driver.find_element_by_css_selector(element).click()
        elif command.startswith("type "):
            _, element, text = command.split(" ", 2)
            self.driver.find_element_by_css_selector(element).send_keys(text)
        elif command == "screenshot":
            screenshot = self.driver.get_screenshot_as_png()
            with open("screenshot.png", "wb") as f:
                f.write(screenshot)
            self.chat_output.append("Screenshot saved as screenshot.png")
        elif command.startswith("record "):
            duration = int(command.split(" ", 1)[1])
            self.record_actions(duration)
        elif command.startswith("playback "):
            filename = command.split(" ", 1)[1]
            self.playback_actions(filename)
        else:
            self.chat_output.append("Unknown command")

    def record_actions(self, duration):
        self.chat_output.append(f"Recording actions for {duration} seconds...")
        actions = []
        start_time = time.time()

        def record():
            while time.time() - start_time < duration:
                x, y = pyautogui.position()
                actions.append((time.time() - start_time, x, y))
                time.sleep(0.1)
            with open("actions.txt", "w") as f:
                for action in actions:
                    f.write(f"{action[0]} {action[1]} {action[2]}\n")
            self.chat_output.append("Recording saved as actions.txt")

        threading.Thread(target=record).start()

    def playback_actions(self, filename):
        self.chat_output.append(f"Playing back actions from {filename}...")
        with open(filename, "r") as f:
            actions = [line.strip().split() for line in f]
            actions = [(float(t), int(x), int(y)) for t, x, y in actions]

        def playback():
            start_time = time.time()
            for action in actions:
                while time.time() - start_time < action[0]:
                    time.sleep(0.01)
                pyautogui.moveTo(action[1], action[2])
                pyautogui.click()

        threading.Thread(target=playback).start()


def main():
    app = QtWidgets.QApplication(sys.argv)
    window = BrowserBot()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
