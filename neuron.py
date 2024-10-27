import numpy as np

from typing import List, Optional


class Neuron:
  input: Optional[float] = None
  output: Optional[float] = None
  delta: Optional[float] = None
  weights: np.ndarray
  bias: float

  def __init__(
      self,
      id: int,
      layer: int,
      input_nodes: List['Neuron']
  ) -> None:
    np.random.seed(1)

    self.id = id
    self.layer = layer
    self.input_nodes = input_nodes

    # bias is a single random value that acts as a constant term
    # added to the input. Biases help the neuron output non-zero
    # values even if the input is zero.
    self.bias = float(np.random.randn())

    # Initialize the weights with random values for each neuron
    # to connect it to the neurons in the previous layer.
    
    # Only initialize weights if there are input nodes
    if input_nodes:
        self.weights = np.random.randn(len(input_nodes))
    else:
        self.weights = np.array([])  # Empty weights for input layer neurons

  # A sigmoid function squashes the input to a value between
  # 0 and 1. This introduces non-linearity to the model, which
  # is essential for learning complex patterns instead of just
  # linear relationships.
  def activation(self, input: float) -> float:
    return 1 / (1 + np.exp(-input))

  # Calculate the derivative of the sigmoid function, which
  # helps adjust the weights based on how much each neuron
  # contributed to the error.
  def sigmoid_derivative(self) -> float:
    assert self.output is not None, "Output must be computed before derivative."
    return self.output * (1 - self.output)

  def forward(self, inputs: np.ndarray) -> None:
    # Compute weighted sum of inputs and add the bias
    self.input = np.dot(self.weights, inputs) + self.bias
    self.output = self.activation(self.input)

    return self.output

  # The delta, used for back-propagation, is the error term
  # for each neuron.
  def calculate_delta(
      self,
      target: Optional[float] = None,
      downstream_deltas: Optional[np.ndarray] = None,
      downstream_weights: Optional[np.ndarray] = None
  ) -> None:
    # If this is the output layer...
    if target is not None:
      # Calculate how far the output is from the target value.
      error = target - self.output
    # If this is a hidden layer...
    elif downstream_deltas is not None and downstream_weights is not None:
      # Calculate error based on delta from neurons in the
      # next layer.
      error = np.dot(downstream_weights, downstream_deltas)
    else:
      raise ValueError(
          "Either target or downstream deltas and weights must be provided.")

    self.delta = error * self.sigmoid_derivative()

  # Update the weights and bias of the neuron based on the
  # delta calculated in the previous step. Each weight is
  # adjusted by a small amount in the direction that reduces
  # the error. The bias also adjusts based on the delta, so it
  # learns along with the weights.
  def update_weights(self, learning_rate: float) -> None:
    assert self.delta is not None, "Delta must be computed before updating weights."
    self.weights += learning_rate * self.delta * \
        np.array([node.output for node in self.input_nodes])
    self.bias += learning_rate * self.delta
