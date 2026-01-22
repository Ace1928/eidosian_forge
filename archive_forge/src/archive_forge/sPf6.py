import tkinter as tk
from tkinter import (
    filedialog,
    messagebox,
    ttk,
    colorchooser,
    simpledialog,
    scrolledtext,
    Spinbox,
    Canvas,
    Checkbutton,
    Entry,
    Frame,
    Label,
    Listbox,
    Menu,
    Menubutton,
    Message,
    Radiobutton,
    Scale,
    Scrollbar,
    Text,
    Toplevel,
    LabelFrame,
    PanedWindow,
    Button,
    OptionMenu,
    PhotoImage,
    BitmapImage,
)
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


class UniversalGUIBuilder:
    """
    A class designed to construct and manage a GUI dynamically with advanced features like drag-and-drop,
    runtime configuration, and manipulation of widgets. Supports saving and loading configurations to JSON files.
    """

    def __init__(self, master: tk.Tk):
        """
        Initialize the UniversalGUIBuilder with a master window and setup the initial GUI components.

        :param master: The main tkinter window.
        :type master: tk.Tk
        """
        self.master = master
        self.master.title("Universal GUI Builder")
        self.master.geometry("800x600")  # Default window size

        self.canvas = tk.Canvas(self.master, bg="white")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.config = {}
        self.widget_icons = {
            "button": "🔘",
            "label": "🏷️",
            "entry": "🔤",
            "checkbox": "☑️",
            "radiobutton": "🔘",
            "listbox": "📋",
            "scale": "📏",
            "frame": "🖼️",
            "canvas": "🎨",
            "text": "📝",
            "menu": "📜",
            "notebook": "📓",
            "panedwindow": "🪟",
            "spinbox": "🔢",
            "scrolledtext": "📜",
            "progressbar": "📊",
            "slider": "🎚️",
            "tab": "📑",
            "combobox": "🔽",
            "tree": "🌲",
            "calendar": "📅",
            "toolbar": "🛠️",
            "statusbar": "📊",
            "dialog": "💬",
            "label_frame": "🖼️",
            "paned_window": "🪟",
            "scrollbar": "📜",
            "message": "💬",
            "menubutton": "🔻",
            "top_level": "🔝",
            "photo_image": "🖼️",
            "bitmap_image": "🖼️",
            "options_menu": "🔽",
        }

        self.setup_menus()
        self.setup_drag_and_drop()

    def setup_menus(self):
        self.menu_bar = tk.Menu(self.master)
        self.master.config(menu=self.menu_bar)

        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(
            label="Save Configuration", command=self.save_configuration
        )
        self.file_menu.add_command(
            label="Load Configuration", command=self.load_configuration
        )
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.add_widgets_menu = tk.Menu(self.menu_bar, tearoff=0)
        for widget_type, icon in self.widget_icons.items():
            self.add_widgets_menu.add_command(
                label=f"{icon} {widget_type.capitalize()}",
                command=lambda w_type=widget_type: self.add_widget(w_type),
            )
        self.menu_bar.add_cascade(label="Add Widgets", menu=self.add_widgets_menu)

        self.preview_button = tk.Button(
            self.frame, text="Preview GUI", command=self.preview_gui
        )
        self.preview_button.pack(side=tk.BOTTOM, pady=10)

    def setup_drag_and_drop(self):
        for widget_type, icon in self.widget_icons.items():
            label = tk.Label(self.canvas, text=icon, font=("Arial", 24))
            label.bind(
                "<Button-1>",
                lambda event, w_type=widget_type: self.drag_start(event, w_type),
            )
            label.bind("<B1-Motion>", self.drag_motion)
            label.pack(side=tk.LEFT, padx=10)

    def drag_start(self, event, widget_type):
        self.drag_data = {"widget_type": widget_type, "x": event.x, "y": event.y}

    def drag_motion(self, event):
        delta_x = event.x - self.drag_data["x"]
        delta_y = event.y - self.drag_data["y"]
        event.widget.move(delta_x, delta_y)
        self.drag_data["x"] = event.x
        self.drag_data["y"] = event.y

    def add_widget(self, widget_type: str):
        """
        Opens a new window to add a widget of the specified type to the GUI configuration, covering all possible
        tkinter widgets and properties.

        :param widget_type: The type of widget to add.
        :type widget_type: str
        """
        new_window = tk.Toplevel(self.master)
        new_window.title(f"Add {widget_type.capitalize()}")

        label = tk.Label(new_window, text=f"Enter {widget_type} properties:")
        label.pack(side=tk.TOP, pady=10)

        name_label = tk.Label(new_window, text="Name:")
        name_label.pack(side=tk.TOP, pady=5)

        name_entry = tk.Entry(new_window)
        name_entry.pack(side=tk.TOP, pady=5)

        command_entry = None
        if widget_type in ["button", "radiobutton", "checkbox", "menu"]:
            command_label = tk.Label(new_window, text="Command (function name):")
            command_label.pack(side=tk.TOP, pady=5)

            command_entry = tk.Entry(new_window)
            command_entry.pack(side=tk.TOP, pady=5)

        options_frame = tk.Frame(new_window)
        options_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        if widget_type in [
            "listbox",
            "scale",
            "entry",
            "text",
            "spinbox",
            "scrolledtext",
            "progressbar",
            "slider",
            "combobox",
        ]:
            options_label = tk.Label(options_frame, text="Options (comma-separated):")
            options_label.pack(side=tk.TOP, pady=5)

            options_entry = tk.Entry(options_frame)
            options_entry.pack(side=tk.TOP, pady=5)

        if widget_type == "canvas":
            width_label = tk.Label(options_frame, text="Width:")
            width_label.pack(side=tk.LEFT, padx=5)

            width_entry = tk.Entry(options_frame, width=5)
            width_entry.pack(side=tk.LEFT, padx=5)

            height_label = tk.Label(options_frame, text="Height:")
            height_label.pack(side=tk.LEFT, padx=5)

            height_entry = tk.Entry(options_frame, width=5)
            height_entry.pack(side=tk.LEFT, padx=5)

        if widget_type == "frame":
            relief_label = tk.Label(options_frame, text="Relief:")
            relief_label.pack(side=tk.LEFT, padx=5)

            relief_options = tk.StringVar()
            relief_options.set("flat")
            relief_dropdown = tk.OptionMenu(
                options_frame,
                relief_options,
                "flat",
                "raised",
                "sunken",
                "groove",
                "ridge",
            )
            relief_dropdown.pack(side=tk.LEFT, padx=5)

        save_button = tk.Button(
            new_window,
            text="Save",
            command=lambda: self.save_widget(
                new_window,
                widget_type,
                name_entry.get(),
                command_entry.get() if command_entry else "",
                options_entry.get() if "options_entry" in locals() else "",
                width_entry.get() if "width_entry" in locals() else "",
                height_entry.get() if "height_entry" in locals() else "",
                relief_options.get() if "relief_options" in locals() else "flat",
            ),
        )
        save_button.pack(side=tk.BOTTOM, pady=10)

    def add_complex_plugin(self, plugin_type: str):
        """
        Adds a complex plugin module to the GUI, such as a terminal emulator, visualization window, or browser window.

        :param plugin_type: The type of complex plugin to add.
        :type plugin_type: str
        """
        new_window = tk.Toplevel(self.master)
        new_window.title(f"Add {plugin_type.capitalize()}")

        if plugin_type == "terminal emulator":
            # Placeholder for terminal emulator integration
            messagebox.showinfo("Info", "Terminal Emulator added successfully!")
        elif plugin_type == "visualization window":
            # Placeholder for visualization window integration
            messagebox.showinfo("Info", "Visualization Window added successfully!")
        elif plugin_type == "browser window":
            # Placeholder for browser window integration
            messagebox.showinfo("Info", "Browser Window added successfully!")
        elif plugin_type == "custom plugin":
            # Placeholder for custom plugin integration
            messagebox.showinfo("Info", "Custom Plugin added successfully!")

        self.config["complex_plugins"].append({"type": plugin_type})
        logging.info(f"Complex plugin added: {plugin_type}")

    def save_widget(
        self,
        window: tk.Toplevel,
        widget_type: str,
        name: str,
        command: str = "",
        options: str = "",
        width: str = "",
        height: str = "",
        relief: str = "flat",
    ):
        """
        Saves the widget configuration to the internal state and closes the configuration window.

        :param window: The window to close after saving.
        :type window: tk.Toplevel
        :param widget_type: The type of widget being saved.
        :type widget_type: str
        :param name: The name of the widget.
        :type name: str
        :param command: The command associated with the widget (if applicable).
        :type command: str
        :param options: Additional options for the widget (if applicable).
        :type options: str
        :param width: Width of the widget (if applicable).
        :type width: str
        :param height: Height of the widget (if applicable).
        :type height: str
        :param relief: Relief style of the widget (if applicable).
        :type relief: str
        """
        widget_config = {
            "name": name,
            "options": options,
            "width": width,
            "height": height,
            "relief": relief,
        }
        if command:
            widget_config["command"] = command
        self.config[f"{widget_type}s"].append(widget_config)
        window.destroy()
        logging.info(f"Widget added: {widget_config}")

    def preview_gui(self):
        """
        Generates a preview of the GUI based on the current configuration.
        """
        preview_window = tk.Toplevel(self.master)
        preview_window.title("GUI Preview")

        for widget_type in self.config:
            for widget in self.config[widget_type]:
                if widget_type == "buttons":
                    tk.Button(
                        preview_window,
                        text=widget["name"],
                        command=lambda w=widget: print(f"Executing {w['command']}"),
                    ).pack(pady=5)
                elif widget_type == "labels":
                    tk.Label(preview_window, text=widget["name"]).pack(pady=5)
                elif widget_type == "entries":
                    tk.Entry(preview_window).pack(pady=5)
                elif widget_type == "complex_plugins":
                    tk.Label(preview_window, text=f"{widget['type']} added").pack(
                        pady=5
                    )

    def save_configuration(self):
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.config, file, indent=4)
            logging.info(f"Configuration saved to {file_path}")

    def load_configuration(self):
        """
        Loads a GUI configuration from a JSON file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.config = json.load(file)
            logging.info(f"Configuration loaded from {file_path}")
            self.preview_gui()


def main():
    root = tk.Tk()
    app = UniversalGUIBuilder(root)
    root.mainloop()


if __name__ == "__main__":
    main()
