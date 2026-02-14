import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from convkan import ConvKAN
from tqdm import tqdm
import logging
import sys
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import ImageTk, Image

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConvKANModel(nn.Module):
    """
    A neural network model based on ConvKAN for image classification.
    """
    def __init__(self, in_channels: int = 1, out_channels1: int = 32, out_channels2: int = 64, 
                 kernel_size: int = 3, fc1_out: int = 128, fc2_out: int = 10) -> None:
        """
        Initializes the ConvKANModel.

        Args:
            in_channels (int): Number of input channels.
            out_channels1 (int): Number of output channels for the first ConvKAN layer.
            out_channels2 (int): Number of output channels for the second ConvKAN layer.
            kernel_size (int): Size of the convolutional kernel.
            fc1_out (int): Number of output units for the first fully connected layer.
            fc2_out (int): Number of output units for the second fully connected layer.
        """
        super(ConvKANModel, self).__init__()
        self.conv1 = ConvKAN(in_channels=in_channels, out_channels=out_channels1, kernel_size=kernel_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = ConvKAN(in_channels=out_channels1, out_channels=out_channels2, kernel_size=kernel_size)
        self.fc1 = nn.Linear(out_channels2 * 5 * 5, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.reshape(-1, self.fc1.in_features)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model(model: ConvKANModel, criterion: nn.Module, optimizer: torch.optim.Optimizer, 
                trainloader: torch.utils.data.DataLoader, epochs: int, progress_bar: ttk.Progressbar, 
                total_progress: ttk.Progressbar, loss_plot: plt.Figure, image_label: tk.Label, root: tk.Tk, device: torch.device) -> None:
    """
    Trains the ConvKAN model.

    Args:
        model (ConvKANModel): The ConvKAN model to train.
        criterion (nn.Module): The loss function.
        optimizer (torch.optim.Optimizer): The optimizer.
        trainloader (torch.utils.data.DataLoader): The data loader for training data.
        epochs (int): The number of training epochs.
        progress_bar (ttk.Progressbar): The progress bar for current epoch progress.
        total_progress (ttk.Progressbar): The progress bar for total training progress.
        loss_plot (plt.Figure): The figure to plot the training loss.
        image_label (tk.Label): The label to display the current training image.
        root (tk.Tk): The root window.
        device (torch.device): The device to run the training on.
    """
    model.to(device)
    losses = []
    for epoch in range(epochs):
        running_loss = 0.0
        progress_bar["value"] = 0
        total_progress["value"] = (epoch / epochs) * 100
        for i, (images, labels) in enumerate(tqdm(trainloader, desc=f"Epoch {epoch+1}", leave=False, file=sys.stdout)):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            progress_bar["value"] = ((i+1) / len(trainloader)) * 100
            root.update_idletasks()

            # Display the current training image
            img = images[0].cpu().squeeze().numpy()
            img = (img * 0.5) + 0.5  # Unnormalize the image
            img = Image.fromarray((img * 255).astype('uint8'))
            img = ImageTk.PhotoImage(img)
            image_label.configure(image=img)
            image_label.image = img

        avg_loss = running_loss / len(trainloader)
        losses.append(avg_loss)
        logger.info(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}")
        total_progress["value"] = ((epoch + 1) / epochs) * 100

        # Update the loss plot
        loss_plot.clear()
        ax = loss_plot.add_subplot(111)
        ax.plot(range(1, len(losses) + 1), losses)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        loss_plot.canvas.draw()

def predict_image(model: ConvKANModel, image: torch.Tensor, device: torch.device) -> int:
    """
    Predicts the label of an input image using the trained ConvKAN model.

    Args:
        model (ConvKANModel): The trained ConvKAN model.
        image (torch.Tensor): The input image tensor.
        device (torch.device): The device to run the prediction on.

    Returns:
        int: The predicted label.
    """
    model.to(device)
    image = image.to(device)
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        _, predicted = torch.max(output, 1)
    return int(predicted.item())

def load_model(model: ConvKANModel, file_path: str) -> None:
    """
    Loads the trained model state dictionary from a file.

    Args:
        model (ConvKANModel): The ConvKAN model to load the state dictionary into.
        file_path (str): The path to the file containing the state dictionary.
    """
    model.load_state_dict(torch.load(file_path))
    model.eval()
    logger.info(f"Loaded model from {file_path}")

def save_model(model: ConvKANModel, file_path: str) -> None:
    """
    Saves the trained model state dictionary to a file.

    Args:
        model (ConvKANModel): The trained ConvKAN model.
        file_path (str): The path to save the state dictionary file.
    """
    torch.save(model.state_dict(), file_path)
    logger.info(f"Saved model to {file_path}")

def start_training(hyperparams: Dict[str, Any], loss_plot: plt.Figure, image_label: tk.Label, 
                   progress_bar: ttk.Progressbar, total_progress: ttk.Progressbar, root: tk.Tk) -> None:
    """
    Starts the model training process.

    Args:
        hyperparams (Dict[str, Any]): The hyperparameters for training.
        loss_plot (plt.Figure): The figure to plot the training loss.
        image_label (tk.Label): The label to display the current training image.
        progress_bar (ttk.Progressbar): The progress bar for current epoch progress.
        total_progress (ttk.Progressbar): The progress bar for total training progress.
        root (tk.Tk): The root window.
    """
    # Get hyperparameters
    epochs = int(hyperparams["epochs"])
    lr = float(hyperparams["lr"])
    in_channels = int(hyperparams["in_channels"])
    out_channels1 = int(hyperparams["out_channels1"])
    out_channels2 = int(hyperparams["out_channels2"])
    kernel_size = int(hyperparams["kernel_size"])
    fc1_out = int(hyperparams["fc1_out"])
    fc2_out = int(hyperparams["fc2_out"])

    # Prepare the dataset
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    # Initialize model, criterion, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvKANModel(in_channels=in_channels, out_channels1=out_channels1, out_channels2=out_channels2, 
                         kernel_size=kernel_size, fc1_out=fc1_out, fc2_out=fc2_out)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Start training
    logger.info("Starting model training...")
    train_model(model, criterion, optimizer, trainloader, epochs, progress_bar, total_progress, loss_plot, image_label, root, device)

    # Save the trained model
    save_path = 'convkan_mnist.pth'
    save_model(model, save_path)
    messagebox.showinfo("Training Complete", f"Model training is complete and saved as '{save_path}'")

def load_pretrained_model() -> None:
    """
    Loads a pretrained model from a file selected by the user.
    """
    file_path = filedialog.askopenfilename()
    if file_path:
        model = ConvKANModel()
        load_model(model, file_path)
        messagebox.showinfo("Model Loaded", "Pretrained model loaded successfully")

def test_single_image(image_label: tk.Label) -> None:
    """
    Tests the trained model on a single random image from the test set.

    Args:
        image_label (tk.Label): The label to display the test image.
    """
    # Load the trained model
    model = ConvKANModel()
    load_model(model, 'convkan_mnist.pth')

    # Get a random test image
    _, testset = torch.utils.data.random_split(torchvision.datasets.MNIST(root='./data', train=False, download=True, 
                                               transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])), 
                                               [1, 9999])
    test_image, _ = testset[0]

    # Make a prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_label = predict_image(model, test_image, device)
    logger.info(f"Predicted Label for Test Image: {predicted_label}")
    messagebox.showinfo("Test Result", f"Predicted Label for Test Image: {predicted_label}")

    # Display the test image
    img = test_image.squeeze().numpy()
    img = (img * 0.5) + 0.5  # Unnormalize the image
    img = Image.fromarray((img * 255).astype('uint8'))
    img = ImageTk.PhotoImage(img)
    image_label.configure(image=img)
    image_label.image = img

def create_input_field(root: tk.Tk, label_text: str, row: int, default_value: str, 
                       tooltip_text: str) -> Tuple[tk.Label, tk.Entry]:
    """
    Creates an input field with a label and tooltip.

    Args:
        root (tk.Tk): The root window.
        label_text (str): The text for the label.
        row (int): The row number for grid layout.
        default_value (str): The default value for the entry field.
        tooltip_text (str): The tooltip text to display on hover.

    Returns:
        Tuple[tk.Label, tk.Entry]: The label and entry field widgets.
    """
    label = tk.Label(root, text=label_text)
    label.grid(row=row, column=0, sticky='e')
    entry = tk.Entry(root)
    entry.insert(0, default_value)
    entry.grid(row=row, column=1)

    tooltip = None

    def show_tooltip(event: tk.Event) -> None:
        """
        Displays a tooltip with the provided text.

        Args:
            event (tk.Event): The event that triggered the tooltip.
        """
        nonlocal tooltip
        tooltip = tk.Toplevel(root)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry(f"+{event.x_root+10}+{event.y_root+10}")
        tooltip_label = tk.Label(tooltip, text=tooltip_text, justify='left', background="#ffffff", relief='solid', borderwidth=1)
        tooltip_label.pack()

    def hide_tooltip(event: tk.Event) -> None:
        """
        Hides the tooltip.

        Args:
            event (tk.Event): The event that triggered the hiding of the tooltip.
        """
        nonlocal tooltip
        if tooltip:
            tooltip.destroy()
            tooltip = None

    entry.bind('<Enter>', show_tooltip)
    entry.bind('<Leave>', hide_tooltip)

    return label, entry

def initialize_gui() -> None:
    """
    Initializes the main GUI window and its components.
    """
    root = tk.Tk()
    root.title("ConvKAN Model Trainer")

    # Create input fields with tooltips
    input_fields = [
        ("Epochs:", 0, "10", "Number of training epochs. Typical values: 10-50."),
        ("Learning Rate:", 1, "0.001", "Learning rate for optimizer. Typical values: 0.1, 0.01, 0.001."),
        ("Input Channels:", 2, "1", "Number of input channels. Default: 1 for grayscale images."),
        ("Conv1 Out Channels:", 3, "32", "Number of output channels for the first convolutional layer. Typical values: 16, 32, 64."),
        ("Conv2 Out Channels:", 4, "64", "Number of output channels for the second convolutional layer. Typical values: 32, 64, 128."),
        ("Kernel Size:", 5, "3", "Size of the convolutional kernel. Typical values: 3, 5, 7."),
        ("FC1 Output:", 6, "128", "Number of output units for the first fully connected layer. Typical values: 64, 128, 256."),
        ("FC2 Output:", 7, "10", "Number of output units for the second fully connected layer. Should match the number of classes.")
    ]

    entries = {}
    for label_text, row, default_value, tooltip_text in input_fields:
        key = label_text.split()[0].lower().replace(":", "")
        label, entry = create_input_field(root, label_text, row, default_value, tooltip_text)
        entries[key] = entry

    # Create progress bars
    progress_bar = ttk.Progressbar(root, length=200, mode='determinate')  
    progress_bar.grid(row=8, columnspan=2, pady=10)

    total_progress = ttk.Progressbar(root, length=200, mode='determinate')
    total_progress.grid(row=9, columnspan=2, pady=10)

    # Create loss plot
    loss_plot_frame = tk.Frame(root)
    loss_plot_frame.grid(row=0, column=2, rowspan=8, padx=20, pady=10)
    loss_plot = plt.Figure(figsize=(5, 4), dpi=100)
    loss_canvas = FigureCanvasTkAgg(loss_plot, master=loss_plot_frame)
    loss_canvas.get_tk_widget().pack()

    # Create image display label
    image_frame = tk.Frame(root)
    image_frame.grid(row=8, column=2, rowspan=4, padx=20, pady=10)
    image_label = tk.Label(image_frame)
    image_label.pack()

    # Create buttons
    train_button = tk.Button(root, text="Start Training", command=lambda: start_training({
        "epochs": entries["epochs"].get(),
        "lr": entries["learning"].get(),
        "in_channels": entries["input"].get(),
        "out_channels1": entries["conv1"].get(),
        "out_channels2": entries["conv2"].get(),
        "kernel_size": entries["kernel"].get(),
        "fc1_out": entries["fc1"].get(),
        "fc2_out": entries["fc2"].get()
    }, loss_plot, image_label, progress_bar, total_progress, root))
    train_button.grid(row=10, column=0, pady=10)

    load_button = tk.Button(root, text="Load Pretrained Model", command=load_pretrained_model)  
    load_button.grid(row=10, column=1, pady=10)

    test_button = tk.Button(root, text="Test Single Image", command=lambda: test_single_image(image_label))
    test_button.grid(row=11, columnspan=2, pady=10)

    # Start the GUI event loop
    root.mainloop()

# Initialize the GUI
initialize_gui()
