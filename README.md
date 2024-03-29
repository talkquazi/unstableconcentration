# Unstable Concentration
A Novel Approach to Noise Generation using a Feed Forward Seed Based Noise algorithm

**`Unstable Concentration`** offers a fresh perspective on noise generation, utilizing advanced techniques to produce RGBA noise with the aid of artificial intelligence (AI). 

This project presents an innovative algorithm grounded in **`1-bit quantization`**, which delivers intricate noise patterns reminiscent of the renowned Perlin Noise method.

With **`Unstable Concentration`**, users can explore a unique blend of AI-driven algorithms and seed-based reproduction to generate consistent and reproducible noise outputs. 

While it may not be the first of its kind, Unstable Concentration offers a refined approach to noise generation. Noise generation method is reproducible with a `seed`, allowing for **consistent** and **predictable** results.

Discover the capabilities of Unstable Concentration and unlock the potential of AI-driven RGBA noise for your projects.
![Alt text](samples/gradio_app.png)

## Features

- Utilizes a 1-bit feedforward neural network to generate RGBA noise.

- Reproducible noise generation with a seed parameter.

- Offers a modern alternative to traditional noise generation methods like Perlin Noise.
  
## Algorithm Overview
The Unstable Concentration algorithm operates as follows:

1.  **Input Tensor Generation**: A random input tensor xx of shape (dimensions*dimensions*32,)(dimensions*dimensions*32,) is created using PyTorch's `torch.randn()` function.
    
2.  **Neural Network Configuration**: An instance of the BitFeedForward class is created with the following parameters:
    
    -   `input_dim`: dimensions * dimensions * 32
    -   `hidden_dim`: dimensions * dimensions * 32
    -   `num_layers`: 4
3.  **Feedforward Propagation**: The BitFeedForward network processes the input tensor xx through four hidden layers, resulting in an output tensor yy.
    
4.  **Reshaping Output**: The output tensor yy is reshaped to dimensions (dimensions,dimensions,32)(dimensions,dimensions,32).
    
5.  **Image Conversion**: The tensor yy is converted to a PIL image using the `Image.fromarray()` function. The values are cast to `uint8` and interpreted as RGBA data.
    
6.  **Image Scaling**: The generated image is optionally scaled up by a multiplier factor specified by the user.
    
7.  **Image Saving**: The resulting image is saved as a PNG file.
    
## Mathematical Formulation
The Unstable Concentration algorithm can be represented mathematically as follows:

1.  **Neural Network Operation**:

```math
y=f(xW1)W2+by=f(xW1​)W2​+b
````

where:

-   yy is the output tensor.
-   xx is the input tensor.
-   W1W1​ and W2W2​ are weight matrices of the neural network.
-   bb is the bias term.
-   ff denotes the activation function used in each layer.

result:

2.  **Image Representation**:

The `RGBA image` II is represented as a `3D array` with dimensions (width,height,channels)(width,height,channels), where each channel (R, G, B, A) corresponds to a color component.

## Installation
To install UnstableConcentration, follow these steps:

1. Clone the repository:

bash
-  `git clone https://github.com/talkquazi/unstableconcentration.git`

- Navigate to the project directory:

bash

-  `cd UnstableConcentration`

- Install dependencies:

bash

 1.  `pip install -r requirements.txt`

## Usage
To generate Unstable Concentration noise, execute the provided Python script `unstableconcentration.py` with the following command-line arguments:

```bash
python unstableconcentration.py [outfilename] [seed] [multiplier] [dimensions]
```

-   `outfilename`: The name of the output file (optional, default is 'output').
-   `seed`: The seed value for random number generation (optional, default is -1 for random seed).
-   `multiplier`: The scaling factor for the image (optional, default is 1).
-   `dimensions`: The dimension of the unstable concentration noise output (optional, default is 4).


*Note: Setting dimensions above 6 requires at minimum 32gb of ram.

## Example

```bash
python unstableconcentration.py my_image 12345 2 4
```

This command generates Unstable Concentration noise with a seed of 12345 and scales the resulting image by a factor of 2 and a dimension of 4x4.

## Gradio

```bash
python gradio_app.py
```

Then visit [localhost:7860](https://localhost:7860/)

## How It Works
UnstableConcentration employs a 1-bit feedforward neural network to generate RGBA noise. The network architecture consists of an input layer with `dimension` units, 4 hidden layers with `dimension` units each, and an output layer with 32 units for color bits.

The network takes a random input tensor as its input and applies a series of transformations to produce the output tensor, which represents RGBA noise. The output tensor is reshaped into a suitable format for image representation and converted to a PIL image.
  
## Acknowledgements
 [BitNet](https://github.com/kyegomez/BitNet) - For creating the 1bit inference interface.

 [ChatGPT](https://chat.openai.com) - For help writing up the algorithms.

 [Github Copilot](https://github.com/features/copilot) - For help with code documentation and formatting outputs.

 
## Contribution
Contributions to UnstableConcentration are welcome! If you'd like to contribute, please follow these guidelines:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes thoroughly.
5. Create a pull request.

## License
`Unstable Concentration` is licensed under the MIT License. See `LICENSE` for more information.
