import torch
from bitnet import BitFeedForward
from PIL import Image
import random
import sys

def process_image(outfilename='output', multiplier=128, seed=-1):
    """
    Process an image using the Unstable Concentration Feed Forward Seed Based Noise algorithm.

    Args:
        outfilename (str, optional): The name of the output file. Defaults to 'output'.
        multiplier (int, optional): The scaling factor for the image. Defaults to 1.
        seed (int, optional): The seed value for random number generation. Defaults to -1.

    Returns:
        None
    """
    print("Multiplier: ", multiplier)
    print("Seed: ", seed)
    print("Generating Unstable Concentration Noise With Dimensions:", 4*multiplier, "x", 4*multiplier, "y")

    # Set the seed value or randomly generate a seed value
    if seed != -1:
        torch.manual_seed(seed)
    else:
        # generate a random number between 0 and 999999999999999999999
        seed = random.randint(0, 9999999999999999999)
        print("Seed value not provided. Generating a random seed value:", seed)
        torch.manual_seed(seed)

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

    # Save the non-scaled image as a PNG file
    image.save(outfilename+ '_'+ str(seed) +'.png')

    # Scale up the image by a multiplier
    new_size = (image.width * multiplier, image.height * multiplier)
    scaled_image = image.resize(new_size)

    # Save the scaled image as a PNG file
    scaled_image.save(outfilename + '_' + str(seed) + '_scaled_'+ str(multiplier) +'.png')

# do main
if __name__ == "__main__":
    # Set the output filename
    outfilename = sys.argv[1] if len(sys.argv) > 1 else 'output'

    # Set the seed value (-1 random) or (Max 9999999999999999999 or 9 Quintillion (10^18))
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    # Set the multiplier value (Think console bits 4bit * multiplier)
    multiplier = int(sys.argv[3]) if len(sys.argv) > 3 else 128

    # Call the process_image function with the seed and multiplier values
    print("Calling the Unstable Concentration process_image function")
    process_image(outfilename, multiplier, seed)
