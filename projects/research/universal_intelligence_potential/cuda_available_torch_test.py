import torch
import pyopencl as cl
import vulkan as vk
from OpenGL.GL import glGetString, GL_RENDERER, GL_VERSION
from OpenGL.GLUT import glutInit, glutCreateWindow, glutDisplayFunc, glutMainLoop
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from torch import nn, optim
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def init_opengl_context():
    try:
        glutInit()
        glutCreateWindow("OpenGL Context")
        glutDisplayFunc(lambda: None)  # Dummy display function
        logging.info("OpenGL context initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize OpenGL context: {e}")

def get_opencl_info():
    try:
        platforms = cl.get_platforms()
        for platform in platforms:
            devices = platform.get_devices()
            for device in devices:
                logging.info(f"OpenCL Device: {device.name}")
    except Exception as e:
        logging.error(f"Failed to get OpenCL information: {e}")

def get_vulkan_info():
    try:
        app_info = vk.VkApplicationInfo(
            sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            pApplicationName="Hello Vulkan",
            applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            pEngineName="No Engine",
            engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
            apiVersion=vk.VK_API_VERSION_1_0,
        )

        create_info = vk.VkInstanceCreateInfo(
            sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            pApplicationInfo=app_info,
        )

        instance = vk.vkCreateInstance(create_info, None)
        physical_devices = vk.vkEnumeratePhysicalDevices(instance)
        for device in physical_devices:
            properties = vk.vkGetPhysicalDeviceProperties(device)
            logging.info(f"Vulkan Device: {properties.deviceName}")
    except Exception as e:
        logging.error(f"Failed to get Vulkan information: {e}")

def get_opengl_info():
    try:
        init_opengl_context()
        opengl_renderer = glGetString(GL_RENDERER).decode('utf-8')
        opengl_version = glGetString(GL_VERSION).decode('utf-8')
        logging.info(f"OpenGL Renderer: {opengl_renderer}")
        logging.info(f"OpenGL Version: {opengl_version}")
    except Exception as e:
        logging.error(f"Failed to get OpenGL information: {e}")

def check_cuda_and_train():
    try:
        cuda_available = torch.cuda.is_available()
        logging.info(f"CUDA Available: {cuda_available}")
        if cuda_available:
            # cuda_device_name = torch.cuda.get_device_name(0)
            #logging.info(f"CUDA Device: {cuda_device_name}")

            class SimpleNN(nn.Module):
                def __init__(self):
                    super(SimpleNN, self).__init__()
                    self.fc = nn.Linear(28 * 28, 10)

                def forward(self, x):
                    x = x.view(-1, 28 * 28)
                    x = self.fc(x)
                    return x

            transform = transforms.Compose([transforms.ToTensor()])
            trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

            model = SimpleNN().cuda()
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            epochs = 1
            for epoch in range(epochs):
                running_loss = 0
                for images, labels in trainloader:
                    images, labels = images.cuda(), labels.cuda()
                    optimizer.zero_grad()
                    output = model(images)
                    loss = criterion(output, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                logging.info(f"Epoch {epoch+1} - Loss: {running_loss/len(trainloader)}")

            testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
            testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
            dataiter = iter(testloader)
            images, labels = dataiter.next()
            images = images.cuda()

            with torch.no_grad():
                output = model(images)
            _, preds = torch.max(output, 1)

            fig = plt.figure(figsize=(12, 12))
            for idx in np.arange(16):
                ax = fig.add_subplot(4, 4, idx+1, xticks=[], yticks=[])
                plt.imshow(images[idx].cpu().numpy().squeeze(), cmap='gray')
                ax.set_title(f"Pred: {preds[idx].item()}")
            plt.show()

        else:
            logging.warning("No CUDA devices found.")
    except Exception as e:
        logging.error(f"Failed to get CUDA information: {e}")

def main():
    get_opencl_info()
    get_vulkan_info()
    get_opengl_info()
    check_cuda_and_train()

if __name__ == "__main__":
    main()
