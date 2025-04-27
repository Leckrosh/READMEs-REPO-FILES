# MLP-From-Scratch

A C++ implementation of a Multilayer Perceptron (MLP) NN adpation of [MLP-From-Scratch](https://github.com/muchlakshay/MLP-From-Scratch) from [muchlakshay](https://github.com/muchlakshay) using Eigen, supporting multiple activation and loss functions, mini-batch gradient descent, and backpropagation for training.

## Features

- **Customizable Architecture**: You can define the number of layers and the number of neurons per layer.
- **Multiple Activation Functions**: Supports ReLU, Sigmoid, Tanh, Leaky Relu and Softmax for non-linearity.
- **Loss Functions**: Supports Mean Squared Error (MSE), Binary Cross Entropy and Cross-Entropy loss functions for regression and classification tasks.
- **Backpropagation**: Implements the backpropagation algorithm for training the network.
- **Mini-batch Gradient Descent**: Optimizes the network using mini-batch gradient descent, allowing for more efficient training on large datasets.
- **Modular Design**: The code is organized in a way that allows for easy extension and modification (e.g., adding new activation functions, loss functions, etc.).

## Prerequisites

- C++17 or later
- [Eigen 3.4.0](https://eigen.tuxfamily.org/) (header-only C++ library for linear algebra, already on the project)

## Getting Started

### Clone the repository

```bash
git clone https://{bitbucket_username}/fx-dev1/mlp_scratch.git
cd MLP-From-Scratch
```
### Compile and run

### Visual Studio 2019

Use VS2019 with C++ 17 or later and open the [VS Solution](./MLP.sln).

The project is already prepared to compile on Debug and Release mode for x86 ARCH.

###  Other possible C++ compilers (Not tested by [Leckrosh](https://github.com/Leckrosh))

Use a C++ compiler that supports C++11 or later (e.g., g++, clang++):

```bash
g++  main.cpp NeuralNetwork.cpp -o NN -I/path/to/eigen
./NN
```
Replace /path/to/eigen with the actual path where Eigen is installed or located.

## Training
- The network can be trained on your own dataset. You can define the input data (features) and target data (labels) in the form of Eigen MatrixXd by loading csv data file using "loadcsv.h" header file, specify the number of epochs, mini-batch size, and learning rate.
```cpp
NeuralNetwork mlp({784, 16, 16, 10}, {"relu", "relu", "softmax"});  // 784 input neurons, 2 hidden layer with 16 neurons and 10 output neuron
mlp.learn(input_data, labels, 1000, 32, 0.01, "cross_entropy_loss", true); // input data, labels, batch size, learning rate, loss function, verbose (true by default)
```
## Test Run On MNSIT 
![Captura de pantalla 2025-04-26 203019](https://github.com/user-attachments/assets/87721831-cb23-4f7f-8c8c-5836c9eb1f79)

- 95% accuracy on MNIST handwritten digit classification with just 20 epochs


## More information

[Leckrosh][https://github.com/Leckrosh]
