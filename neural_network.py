import numpy as np

from typing import List
from neuron import Neuron


class NeuralNetwork:
  def __init__(
      self,
      node_layer_counts: List[int],
      output_threshold: float = 0.5
  ) -> None:
    self.layers: List[List[Neuron]] = []
    self.learning_rate = 0.1
    self.output_threshold = output_threshold

    self.build_layers(node_layer_counts)

  def build_layer(
      self,
      node_count: int,
      layer_id: int,
      previous_layer: List[Neuron]
  ) -> List[Neuron]:
    return [Neuron(node_id, layer_id, previous_layer)
            for node_id in range(1, node_count + 1)]

  def build_layers(self, node_layer_counts: List[int]) -> None:
    for layer_id, count in enumerate(node_layer_counts):
      assert count > 0, "Each layer must have at least one neuron."

      # The first layer is the input layer, which has no weights.
      if layer_id == 0:
        self.layers.append(self.build_layer(
            count, layer_id + 1, []))

      # The middle layers are hidden layers. The last layer is the output layer.
      else:
        self.layers.append(self.build_layer(
            count, layer_id + 1, self.layers[layer_id - 1]))

  def forward(self, inputs: np.ndarray) -> np.ndarray:
    for layer_index, layer in enumerate(self.layers):
      if (layer_index == 0):
        # Input layer: Directly pass inputs as outputs
        for i, neuron in enumerate(layer):
          neuron.output = inputs[i]
      else:
        inputs = np.array([neuron.forward(inputs) for neuron in layer])

    # Return the output of the last layer
    return inputs

  def backpropagate(self, outputs: np.ndarray) -> None:
    for layer_index in reversed(range(len(self.layers))):
      layer = self.layers[layer_index]

      # Output layer
      if layer_index == len(self.layers) - 1:
        for node_index, neuron in enumerate(layer):
          neuron.calculate_delta(target=outputs[node_index])

      # Hidden layers
      else:
        next_layer = self.layers[layer_index + 1]
        downstream_deltas = np.array([neuron.delta for neuron in next_layer])
        downstream_weights_all = np.array(
            [neuron.weights for neuron in next_layer])

        for neuron_index, neuron in enumerate(layer):
          # Get the downstream weights from this neuron to the
          # neurons in the next layer.
          downstream_weights = downstream_weights_all[:, neuron_index]

          neuron.calculate_delta(downstream_deltas=downstream_deltas,
                                 downstream_weights=downstream_weights)

  def update_weights(self) -> None:
    for layer in self.layers[1:]:  # Skip the input layer
      for neuron in layer:
        neuron.update_weights(self.learning_rate)

  def train(
      self,
      training_inputs: List[np.ndarray],
      training_outputs: List[np.ndarray],
      training_iterations: int
  ) -> None:
    print("Training the neural network...")
    
    for iteration in range(training_iterations):
      for inputs, outputs in zip(training_inputs, training_outputs):
        self.forward(np.array(inputs))
        self.backpropagate(np.array(outputs))
        self.update_weights()

      if (iteration + 1) % 1000 == 0:
        percentage_complete = str((iteration + 1) / training_iterations * 100)[0:4]
        print(f"Iteration {iteration + 1} complete ({percentage_complete}% done).")

    pass

  def think(self, inputs: List[float]) -> List[float]:
    raw_outputs = self.forward(np.array(inputs))

    # Apply threshold to classify outputs as 0 or 1
    classified_outputs = tuple(
        [1 if output > self.output_threshold else 0 for output in raw_outputs])

    return classified_outputs
