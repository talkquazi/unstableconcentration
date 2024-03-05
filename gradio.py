import torch
from bitnet import BitFeedForward
from PIL import Image
import random
import gradio as gr

def process_image(multiplier=128, seed=-1):
    """
    Process an image using the Unstable Concentration Feed Forward Seed Based Noise algorithm.

    Args:
        multiplier (int, optional): The scaling factor for the image. Defaults to 128.
        seed (int, optional): The seed value for random number generation. Defaults to -1.

    Returns:
        Tuple of PIL images: The original and scaled images.
    """
    # Set the seed value or randomly generate a seed value
    if seed != -1:
        torch.manual_seed(seed)
    else:
        # generate a random number between 0 and 999999999999999999999
        seed = random.randint(0, 9999999999999999999)
        print("Seed value not provided. Generating a random seed value:", int(seed))
        torch.manual_seed(int(seed))

    # Create a random input tensor of shape (512,)
    x = torch.randn(512)

    # Create an instance of the BitFeedForward class with the following parameters:
    # - input_dim: 512
    # - hidden_dim: 512
    # - num_layers: 4
    ff = BitFeedForward(512, 512, 4)

    # Apply the BitFeedForward network to the input tensor x
    y = ff(x)

    # Reshape the output tensor y to (4, 4, 32)
    y = y.view(4, 4, 32)

    # Convert the tensor to a PIL image
    image = Image.fromarray(y.detach().numpy().astype('uint8'), 'RGBA')

    # Scale up the image by a multiplier
    new_size = (int(image.width * int(multiplier)), int(image.height * multiplier))
    scaled_image = image.resize(new_size)

    return scaled_image, image

# Create Gradio interface
inputs = [
    gr.inputs.Number(label="Multiplier", default=128),
    gr.inputs.Number(label="Seed", default=-1)
]

def random_seed():
    return random.randint(1, 999999999999999999999)

def update_seed(button):
    if button:
        return random_seed()
    else:
        return -1

outputs = [
    gr.outputs.Image(type="pil", label="Scaled Image"),
    gr.outputs.Image(type="pil", label="Original Image ( 4 pixels by 4 pixels )")
]

title = "Unstable Concentration RGBA Noise Generator"
description = "Generate RGBA noise images using the Unstable Concentration algorithm."

examples = [
    ["128", 9999999999999999999],
    ["96", 9999999999999999999],
    ["64", 9999999999999999999],
    ["32", 9999999999999999999],
    ["16", 9999999999999999999],
    ["8", 9999999999999999999],
    ["4", 9999999999999999999],
    ["2", 9999999999999999999],
    ["1", 9999999999999999999],
    ["128", -1],
    ["96", -1],
    ["64", -1],
    ["32", -1],
    ["16", -1],
    ["8", -1],
    ["4", -1],
    ["2", -1],
    ["1", -1]
]

gr.Interface(process_image, inputs, outputs, title=title, description=description, 
             live=False, capture_session=True, examples=examples).launch()