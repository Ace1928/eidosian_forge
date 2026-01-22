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
        self.master.geometry("1280x720")  # Default window size optimized for usability

        self.canvas = tk.Canvas(self.master, bg="white", width=1260, height=680)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.canvas.bind("<Button-1>", self.canvas_click)

        self.config = {"widgets": [], "complex_plugins": []}
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
        self.setup_widget_properties_panel()

    def setup_menus(self) -> None:
        """
        Setup the menu bar with File and Add Widgets menus.
        """
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

    def setup_drag_and_drop(self) -> None:
        """
        Setup the drag and drop functionality for adding widgets to the canvas.
        """
        self.drag_data = {}
        self.widget_palette = tk.Frame(self.master, bg="lightgray")
        self.widget_palette.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        for widget_type, icon in self.widget_icons.items():
            label = tk.Label(
                self.widget_palette, text=icon, font=("Arial", 24), bg="lightgray"
            )
            label.bind(
                "<ButtonPress-1>",
                lambda event, w_type=widget_type: self.drag_start(event, w_type),
            )
            label.bind(
                "<ButtonRelease-1>",
                lambda event, w_type=widget_type: self.drag_stop(event, w_type),
            )
            label.bind("<B1-Motion>", self.drag_motion)
            label.pack(pady=5)

    def setup_widget_properties_panel(self) -> None:
        """
        Setup the widget properties panel for editing widget properties.
        """
        self.properties_panel = tk.Frame(self.master, bg="lightgray")
        self.properties_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

        self.properties_label = tk.Label(
            self.properties_panel, text="Widget Properties", font=("Arial", 16)
        )
        self.properties_label.pack(pady=10)

        self.properties_frame = tk.Frame(self.properties_panel, bg="lightgray")
        self.properties_frame.pack(fill=tk.BOTH, expand=True)

    def drag_start(self, event: tk.Event, widget_type: str) -> None:
        """
        Start dragging a widget from the widget palette.

        :param event: The tkinter event that triggered the drag start.
        :type event: tk.Event
        :param widget_type: The type of widget being dragged.
        :type widget_type: str
        """
        self.drag_data = {"widget_type": widget_type, "x": event.x, "y": event.y}
        self.dragged_widget = tk.Label(
            self.canvas, text=self.widget_icons[widget_type], font=("Arial", 24)
        )
        self.dragged_widget.place(x=event.x, y=event.y)

    def drag_motion(self, event: tk.Event) -> None:
        """
        Move the dragged widget as the mouse moves.

        :param event: The tkinter event that triggered the drag motion.
        :type event: tk.Event
        """
        if self.dragged_widget:
            self.canvas.move(self.dragged_widget, event.x, event.y)

    def drag_stop(self, event: tk.Event, widget_type: str) -> None:
        """
        Stop dragging a widget and add it to the canvas if dropped on the canvas.

        :param event: The tkinter event that triggered the drag stop.
        :type event: tk.Event
        :param widget_type: The type of widget being dragged.
        :type widget_type: str
        """
        if self.dragged_widget:
            widget_x = self.canvas.winfo_pointerx() - self.canvas.winfo_rootx()
            widget_y = self.canvas.winfo_pointery() - self.canvas.winfo_rooty()

            if (
                0 <= widget_x <= self.canvas.winfo_width()
                and 0 <= widget_y <= self.canvas.winfo_height()
            ):
                self.add_widget(widget_type, widget_x, widget_y)

            self.dragged_widget.destroy()
            self.dragged_widget = None

    def canvas_click(self, event: tk.Event) -> None:
        """
        Handle clicking on the canvas to select a widget.

        :param event: The tkinter event that triggered the canvas click.
        :type event: tk.Event
        """
        x, y = event.x, event.y
        for widget in self.config["widgets"]:
            if (
                widget["x"] <= x <= widget["x"] + widget["width"]
                and widget["y"] <= y <= widget["y"] + widget["height"]
            ):
                self.select_widget(widget)
                break
        else:
            self.deselect_widget()

    def select_widget(self, widget: dict) -> None:
        """
        Select a widget and display its properties in the properties panel.

        :param widget: The widget configuration dictionary.
        :type widget: dict
        """
        self.selected_widget = widget
        self.display_widget_properties()

    def deselect_widget(self) -> None:
        """
        Deselect the currently selected widget and clear the properties panel.
        """
        self.selected_widget = None
        self.clear_properties_panel()

    def display_widget_properties(self) -> None:
        """
        Display the properties of the selected widget in the properties panel.
        """
        self.clear_properties_panel()

        if self.selected_widget:
            for key, value in self.selected_widget.items():
                if key not in ["x", "y", "width", "height"]:
                    label = tk.Label(self.properties_frame, text=f"{key.capitalize()}:")
                    label.pack(anchor=tk.W)

                    entry = tk.Entry(self.properties_frame)
                    entry.insert(tk.END, value)
                    entry.pack(anchor=tk.W)

            save_button = tk.Button(
                self.properties_frame,
                text="Save Properties",
                command=self.save_widget_properties,
            )
            save_button.pack(pady=10)

    def clear_properties_panel(self) -> None:
        """
        Clear the contents of the properties panel.
        """
        for child in self.properties_frame.winfo_children():
            child.destroy()

    def save_widget_properties(self) -> None:
        """
        Save the edited properties of the selected widget.
        """
        if self.selected_widget:
            entries = self.properties_frame.winfo_children()
            for i in range(0, len(entries), 2):
                key = entries[i].cget("text").strip(":")
                value = entries[i + 1].get()
                self.selected_widget[key.lower()] = value

            self.update_widget_config()

    def add_widget(
        self, widget_type: str, x: int = 0, y: int = 0, **kwargs
    ) -> None:
        """
        Add a widget to the canvas and the configuration.

        :param widget_type: The type of widget to add.
        :type widget_type: str
        :param x: The x-coordinate of the widget on the canvas.
        :type x: int
        :param y: The y-coordinate of the widget on the canvas.
        :type y: int
        :param kwargs: Additional keyword arguments for the widget configuration.
        """
        widget_config = {
            "type": widget_type,
            "x": x,
            "y": y,
            "width": 100,
            "height": 30,
            **kwargs,
        }
        self.config["widgets"].append(widget_config)
        self.draw_widget(widget_config)

    def draw_widget(self, widget_config: dict) -> None:
        """
        Draw a widget on the canvas based on its configuration.

        :param widget_config: The widget configuration dictionary.
        :type widget_config: dict
        """
        widget_type = widget_config["type"]
        x, y = widget_config["x"], widget_config["y"]
        width, height = widget_config.get("width", 100), widget_config.get("height", 30)

        if widget_type == "button":
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="lightblue", tags=("widget",)
            )
            self.canvas.create_text(
                x + width // 2,
                y + height // 2,
                text=widget_config.get("text", "Button"),
                tags=("widget",),
            )
        elif widget_type == "label":
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="lightgreen", tags=("widget",)
            )
            self.canvas.create_text(
                x + width // 2,
                y + height // 2,
                text=widget_config.get("text", "Label"),
                tags=("widget",),
            )
        elif widget_type == "entry":
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="white", tags=("widget",)
            )
        elif widget_type == "complex_plugins":
            self.canvas.create_rectangle(
                x, y, x + width, y + height, fill="lightgray", tags=("widget",)
            )
            self.canvas.create_text(
                x + width // 2,
                y + height // 2,
                text=widget_config.get("name", "Complex Plugin"),
                tags=("widget",),
            )

    def update_widget_config(self) -> None:
        """
        Update the configuration of the selected widget based on its current state on the canvas.
        """
        if self.selected_widget:
            widget_id = self.canvas.find_withtag("current")[0]
            coords = self.canvas.coords(widget_id)
            self.selected_widget["x"] = coords[0]
            self.selected_widget["y"] = coords[1]
            self.selected_widget["width"] = coords[2] - coords[0]
            self.selected_widget["height"] = coords[3] - coords[1]

    def save_configuration(self) -> None:
        """
        Save the current GUI configuration to a JSON file.
        """
        file_path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON files", "*.json")]
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.config, file, indent=4)
            logging.info(f"Configuration saved to {file_path}")

    def load_configuration(self) -> None:
        """
        Load a GUI configuration from a JSON file.
        """
        file_path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as file:
                self.config = json.load(file)
            logging.info(f"Configuration loaded from {file_path}")
            self.canvas.delete("all")
            for widget_config in self.config["widgets"]:
                self.draw_widget(widget_config)

    def preview_gui(self) -> None:
        """
        Generate a preview of the GUI based on the current configuration.
        """
        preview_window = tk.Toplevel(self.master)
        preview_window.title("GUI Preview")

        for widget_config in self.config["widgets"]:
            widget_type = widget_config["type"]
            x, y = widget_config["x"], widget_config["y"]
            width, height = widget_config.get("width", 100), widget_config.get(
                "height", 30
            )

            if widget_type == "button":
                tk.Button(
                    preview_window,
                    text=widget_config.get("text", "Button"),
                    command=lambda w=widget_config: print(f"Executing {w['command']}"),
                ).place(x=x, y=y, width=width, height=height)
            elif widget_type == "label":
                tk.Label(
                    preview_window, text=widget_config.get("text", "Label")
                ).place(x=x, y=y, width=width, height=height)
            elif widget_type == "entry":
                tk.Entry(preview_window).place(x=x, y=y, width=width, height=height)
            elif widget_type == "complex_plugins":
                tk.Label(
                    preview_window, text=widget_config.get("name", "Complex Plugin")
                tk.Label(preview_window, text=f"{widget['type']} added").pack(pady=5)


def main():
    root = tk.Tk()
    app = UniversalGUIBuilder(root)
    root.mainloop()


if __name__ == "__main__":
    main()
