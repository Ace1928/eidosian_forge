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
    A class for building a universal GUI using Tkinter, designed to be modular, extensible, and robust.
    This class encapsulates the functionality required to construct a versatile graphical user interface
    with a variety of widgets and custom configurations, adhering to high standards of software engineering.
    """

    def __init__(self, master: tk.Tk) -> None:
        """
        Initialize the Universal GUI Builder with a master window.

        :param master: The main window which acts as the parent for all other widgets.
        :type master: tk.Tk
        """
        self.master: tk.Tk = master
        self.master.title("Universal GUI Builder")
        self.config: dict = {"widgets": []}
        self.canvas: tk.Canvas = tk.Canvas(self.master)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<Button-1>", self.canvas_click)
        self.canvas.bind("<B1-Motion>", self.canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self.canvas_release)
        self.drag_start_x: int = 0
        self.drag_start_y: int = 0
        self.selected_widget: Optional[dict] = None
        self.create_menu()
        self.create_toolbar()
        self.create_properties_panel()

    def create_menu(self) -> None:
        """
        Create the menu bar for the GUI builder, adding essential functionality such as file management,
        editing options, view adjustments, and help information.
        """
        menu_bar: tk.Menu = tk.Menu(self.master)
        self.master.config(menu=menu_bar)

        file_menu: tk.Menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="New", command=self.new_project)
        file_menu.add_command(label="Open", command=self.open_project)
        file_menu.add_command(label="Save", command=self.save_project)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.master.quit)
        menu_bar.add_cascade(label="File", menu=file_menu)

        edit_menu: tk.Menu = tk.Menu(menu_bar, tearoff=0)
        edit_menu.add_command(label="Undo", command=self.undo)
        edit_menu.add_command(label="Redo", command=self.redo)
        menu_bar.add_cascade(label="Edit", menu=edit_menu)

        view_menu: tk.Menu = tk.Menu(menu_bar, tearoff=0)
        view_menu.add_command(label="Zoom In", command=self.zoom_in)
        view_menu.add_command(label="Zoom Out", command=self.zoom_out)
        menu_bar.add_cascade(label="View", menu=view_menu)

        help_menu: tk.Menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="About", command=self.show_about)
        menu_bar.add_cascade(label="Help", menu=help_menu)

    def create_toolbar(self) -> None:
        """
        Create a toolbar for the GUI builder, providing quick access to common actions through buttons,
        each equipped with an icon for visual identification.
        """
        toolbar: tk.Frame = tk.Frame(self.master, bd=1, relief=tk.RAISED)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        button_data: List[Tuple[str, Callable, str]] = [
            ("New", self.new_project, "new.png"),
            ("Open", self.open_project, "open.png"),
            ("Save", self.save_project, "save.png"),
            ("Undo", self.undo, "undo.png"),
            ("Redo", self.redo, "redo.png"),
            ("Zoom In", self.zoom_in, "zoom_in.png"),
            ("Zoom Out", self.zoom_out, "zoom_out.png"),
        ]

        for label, command, image_file in button_data:
            image: tk.PhotoImage = tk.PhotoImage(file=image_file)
            button: tk.Button = tk.Button(toolbar, image=image, command=command)
            button.image = image  # Keep a reference to avoid garbage collection
            button.pack(side=tk.LEFT, padx=2, pady=2)

    def create_properties_panel(self) -> None:
        """
        Create a properties panel for editing the properties of selected widgets.
        This panel allows for detailed configuration of widget attributes.
        """
        properties_panel: tk.Frame = tk.Frame(self.master, bd=1, relief=tk.RAISED)
        properties_panel.pack(side=tk.RIGHT, fill=tk.Y)

        label: tk.Label = tk.Label(properties_panel, text="Properties")
        label.pack(pady=10)

        self.properties_entries: Dict[str, tk.Entry] = {}
        properties: List[str] = [
            "text",
            "width",
            "height",
            "fg",
            "bg",
            "font",
            "command",
            "value",
            "variable",
        ]

        for prop in properties:
            frame: tk.Frame = tk.Frame(properties_panel)
            frame.pack(fill=tk.X, padx=5, pady=5)

            label = tk.Label(frame, text=prop.capitalize())
            label.pack(side=tk.LEFT)

            entry = tk.Entry(frame)
            entry.pack(side=tk.RIGHT, fill=tk.X, expand=True)

            self.properties_entries[prop] = entry

        button = tk.Button(
            properties_panel, text="Apply", command=self.apply_properties
        )
        button.pack(pady=10)

    def canvas_click(self, event):
        """
        Handle the click event on the canvas.
        """
        item = self.canvas.find_closest(event.x, event.y)
        if item:
            self.selected_widget = next(
                (w for w in self.config["widgets"] if w["id"] == item[0]), None
            )
            if self.selected_widget:
                self.show_widget_properties()
                logging.debug(f"Widget selected with ID: {item[0]}")
            else:
                logging.error("No widget found at the clicked position.")
        else:
            self.selected_widget = None
            self.clear_properties_panel()
            logging.debug("Clicked on canvas without hitting a widget.")

    def canvas_drag(self, event):
        """
        Handle the drag event on the canvas.
        """
        if self.selected_widget is None:
            logging.debug("Drag attempted without a selected widget.")
            return
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        self.canvas.move(self.selected_widget["id"], dx, dy)
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        logging.debug(
            f"Widget {self.selected_widget['id']} moved to ({event.x}, {event.y})."
        )

    def canvas_release(self, event):
        """
        Handle the release event on the canvas.
        """
        if self.selected_widget is None:
            logging.debug("Release attempted without a selected widget.")
            return
        self.update_widget_config()
        logging.debug(
            f"Widget {self.selected_widget['id']} configuration updated on release."
        )

    def show_widget_properties(self):
        """
        Show the properties of the selected widget in the properties panel.
        """
        if not self.selected_widget:
            logging.error("Attempt to show properties without a selected widget.")
            return
        for prop, entry in self.properties_entries.items():
            entry.delete(0, tk.END)
            entry.insert(0, self.selected_widget.get(prop, ""))
        logging.info(
            f"Properties displayed for widget ID: {self.selected_widget['id']}"
        )

    def clear_properties_panel(self):
        """
        Clear the properties panel.
        """
        for entry in self.properties_entries.values():
            entry.delete(0, tk.END)
        logging.info("Properties panel cleared.")

    def apply_properties(self):
        """
        Apply the modified properties to the selected widget.
        """
        if not self.selected_widget:
            logging.error("Attempt to apply properties without a selected widget.")
            return
        for prop, entry in self.properties_entries.items():
            value = entry.get()
            if value:
                self.selected_widget[prop] = value
                logging.debug(
                    f"Property {prop} set to {value} for widget ID: {self.selected_widget['id']}"
                )
        self.update_widget_on_canvas()

    def update_widget_on_canvas(self):
        """
        Update the widget on the canvas based on the modified properties.
        """
        if not self.selected_widget:
            logging.error("Attempt to update canvas without a selected widget.")
            return
        widget_id = self.selected_widget["id"]
        self.canvas.delete(widget_id)
        self.create_widget(self.selected_widget)
        logging.info(f"Widget {widget_id} updated on canvas.")

    def create_widget(self, widget_config):
        """
        Create a new widget on the canvas based on the given configuration.
        """
        widget_type = widget_config["type"]
        x, y = widget_config["x"], widget_config["y"]
        width, height = widget_config.get("width", 100), widget_config.get("height", 30)

        if widget_type == "button":
            widget_id = self.canvas.create_window(
                x,
                y,
                window=tk.Button(self.canvas, text=widget_config.get("text", "Button")),
                width=width,
                height=height,
            )
        elif widget_type == "label":
            widget_id = self.canvas.create_window(
                x,
                y,
                window=tk.Label(self.canvas, text=widget_config.get("text", "Label")),
                width=width,
                height=height,
            )
        elif widget_type == "entry":
            widget_id = self.canvas.create_window(
                x, y, window=tk.Entry(self.canvas), width=width, height=height
            )
        elif widget_type == "complex_plugins":
            widget_id = self.canvas.create_window(
                x,
                y,
                window=tk.Label(
                    self.canvas, text=widget_config.get("name", "Complex Plugin")
                ),
                width=width,
                height=height,
            )
        else:
            raise ValueError(f"Unsupported widget type: {widget_type}")

        widget_config["id"] = widget_id
        self.config["widgets"].append(widget_config)

    def update_widget_config(self):
        """
        Update the configuration of the selected widget based on its current position and size.
        """
        if self.selected_widget is None:
            return
        coords = self.canvas.coords(self.selected_widget["id"])
        self.selected_widget["x"] = coords[0]
        self.selected_widget["y"] = coords[1]

    def new_project(self):
        """
        Create a new project by resetting the configuration and clearing the canvas.
        """
        self.config = {"widgets": []}
        self.canvas.delete("all")
        self.selected_widget = None
        self.clear_properties_panel()

    def open_project(self):
        """
        Open an existing project from a file.
        """
        file_path = filedialog.askopenfilename(
            filetypes=[("JSON Files", "*.json")], defaultextension=".json"
        )
        if file_path:
            with open(file_path, "r") as file:
                self.config = json.load(file)
            self.canvas.delete("all")
            for widget_config in self.config["widgets"]:
                self.create_widget(widget_config)

    def save_project(self):
        """
        Save the current project to a file.
        """
        file_path = filedialog.asksaveasfilename(
            filetypes=[("JSON Files", "*.json")], defaultextension=".json"
        )
        if file_path:
            with open(file_path, "w") as file:
                json.dump(self.config, file, indent=2)

    def undo(self):
        """
        Undo the last action.
        """
        pass

    def redo(self):
        """
        Redo the last undone action.
        """
        pass

    def zoom_in(self):
        """
        Zoom in on the canvas.
        """
        pass

    def zoom_out(self):
        """
        Zoom out on the canvas.
        """
        pass

    def show_about(self):
        """
        Show the "About" dialog.
        """
        messagebox.showinfo(
            "About", "Universal GUI Builder\nVersion 1.0\n\nCreated by Your Name"
        )

    def run(self):
        """
        Run the GUI builder application.
        """
        self.master.mainloop()

    def generate_code(self):
        """
        Generate the code for the GUI based on the current configuration.
        """
        code = "import tkinter as tk\n\n"
        code += "class MyGUI:\n"
        code += "    def __init__(self, master):\n"
        code += "        self.master = master\n"

        for widget_config in self.config["widgets"]:
            widget_type = widget_config["type"]
            x, y = widget_config["x"], widget_config["y"]
            width, height = widget_config.get("width", 100), widget_config.get(
                "height", 30
            )

            if widget_type == "button":
                code += f"        tk.Button(master, text='{widget_config.get('text', 'Button')}', command=self.{widget_config.get('command', 'button_click')}).place(x={x}, y={y}, width={width}, height={height})\n"
            elif widget_type == "label":
                code += f"        tk.Label(master, text='{widget_config.get('text', 'Label')}').place(x={x}, y={y}, width={width}, height={height})\n"
            elif widget_type == "entry":
                code += f"        tk.Entry(master).place(x={x}, y={y}, width={width}, height={height})\n"
            elif widget_type == "complex_plugins":
                code += f"        # Add code for {widget_config.get('name', 'Complex Plugin')} here\n"

        code += "\n    def button_click(self):\n"
        code += "        print('Button clicked!')\n"

        code += "\nif __name__ == '__main__':\n"
        code += "    root = tk.Tk()\n"
        code += "    gui = MyGUI(root)\n"
        code += "    root.mainloop()\n"

        return code

    def preview(self):
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
                    command=lambda: print(
                        f"Executing {widget_config.get('command', 'button_click')}"
                    ),
                ).place(x=x, y=y, width=width, height=height)
            elif widget_type == "label":
                tk.Label(preview_window, text=widget_config.get("text", "Label")).place(
                    x=x, y=y, width=width, height=height
                )
            elif widget_type == "entry":
                tk.Entry(preview_window).place(x=x, y=y, width=width, height=height)
            elif widget_type == "complex_plugins":
                tk.Label(
                    preview_window, text=widget_config.get("name", "Complex Plugin")
                ).place(x=x, y=y, width=width, height=height)


def main():
    root = tk.Tk()
    app = UniversalGUIBuilder(root)
    app.run()


if __name__ == "__main__":
    main()
