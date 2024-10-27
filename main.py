import numpy as np

from neural_network import NeuralNetwork

up = [
    0, 1, 0,
    1, 1, 1,
    0, 0, 0
]

down = [
    0, 0, 0,
    1, 1, 1,
    0, 1, 0
]

left = [
    0, 1, 0,
    1, 1, 0,
    0, 1, 0
]

right = [
    0, 1, 0,
    0, 1, 1,
    0, 1, 0
]

training_inputs = [up, down, left, right]

training_outputs = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

output_mapping = {
    (1, 0, 0, 0): "up",
    (0, 1, 0, 0): "down",
    (0, 0, 1, 0): "left",
    (0, 0, 0, 1): "right"
}


if __name__ == "__main__":
  node_layer_counts = [9, 10, 4]
  neural_network = NeuralNetwork(node_layer_counts)

  neural_network.train(training_inputs, training_outputs, 15000)

  user_inputs = str(input("User Inputs: ")).split(" ")
  user_inputs = list(map(float, user_inputs))

  print("Considering input...", user_inputs)

  output = neural_network.think(np.array(user_inputs, dtype=float))
  classified_output = output_mapping.get(output, "unknown")

  if classified_output == "unknown":
    print(f"I'm not sure what the answer is.")
    print(f"The outputs '{output}' don't match a known combination.")
  else:
    print(f"I think the answer is {classified_output}.")
