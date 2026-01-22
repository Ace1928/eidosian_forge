import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import io
import lzma
import hashlib
import sqlite3
import base64
import threading


# Context manager for database connection
class DatabaseConnection:
    def __enter__(self):
        self.conn = sqlite3.connect("image_db.sqlite")
        return self.conn.cursor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.conn.close()


# Initialize database connection and table
def init_db():
    with DatabaseConnection() as cursor:
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS images
                       (hash TEXT PRIMARY KEY, compressed_data BLOB)"""
        )


def resize_image(image, max_size=(800, 600)):
    image.thumbnail(max_size, Image.ANTIALIAS)
    return image


def load_image():
    try:
        file_path = filedialog.askopenfilename()
        if file_path:
            with Image.open(file_path) as img:
                img = resize_image(img.convert("RGBA"))
                img_bytes = io.BytesIO()
                img.save(img_bytes, format="PNG")
                original_image_data = img_bytes.getvalue()

                tk_img = ImageTk.PhotoImage(img)
                original_img_label.config(image=tk_img)
                original_img_label.image = tk_img

                window.original_image_data = original_image_data
                window.img_hash = hashlib.sha256(original_image_data).hexdigest()

                compress_btn.config(state="normal")
                status_label.config(text="Image loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load the image: {e}")


def compress_image():
    try:
        compress_btn.config(state="disabled")
        status_label.config(text="Compressing image...")
        compressed_data = lzma.compress(window.original_image_data)
        with DatabaseConnection() as cursor:
            cursor.execute(
                "INSERT OR REPLACE INTO images (hash, compressed_data) VALUES (?, ?)",
                (window.img_hash, compressed_data),
            )
        status_label.config(text="Image compressed and stored successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to compress or store the image: {e}")
    finally:
        compress_btn.config(state="normal")


def get_decompressed_image(hash_value):
    try:
        with DatabaseConnection() as cursor:
            cursor.execute(
                "SELECT compressed_data FROM images WHERE hash = ?", (hash_value,)
            )
            result = cursor.fetchone()
            if result:
                return lzma.decompress(result[0])
    except Exception as e:
        messagebox.showerror(
            "Error", f"Failed to retrieve or decompress the image: {e}"
        )
        return None


def decompress_and_show():
    try:
        decompressed_data = get_decompressed_image(window.img_hash)
        if decompressed_data:
            img = Image.open(io.BytesIO(decompressed_data))
            tk_img = ImageTk.PhotoImage(img)
            decompressed_img_label.config(image=tk_img)
            decompressed_img_label.image = tk_img
    except Exception as e:
        messagebox.showerror("Error", f"Failed to decompress the image: {e}")


# GUI Design
window = tk.Tk()
window.title("Image Compressor")

load_btn = tk.Button(
    window,
    text="Load Image",
    command=lambda: threading.Thread(target=load_image).start(),
)
load_btn.pack()

compress_btn = tk.Button(
    window,
    text="Compress & Store Image",
    state="disabled",
    command=lambda: threading.Thread(target=compress_image).start(),
)
compress_btn.pack()

decompress_btn = tk.Button(
    window,
    text="Decompress & Show Image",
    command=lambda: threading.Thread(target=decompress_and_show).start(),
)
decompress_btn.pack()

original_img_label = tk.Label(window)
original_img_label.pack(side="left", padx=10)

decompressed_img_label = tk.Label(window)
decompressed_img_label.pack(side="right", padx=10)

status_label = tk.Label(window, text="")
status_label.pack()

init_db()

window.mainloop()
