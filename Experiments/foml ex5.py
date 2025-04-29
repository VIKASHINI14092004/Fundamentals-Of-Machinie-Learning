import numpy as np
class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # initialize weights matrix and biases
        self.W_input_hidden = np.random.rand(self.input_size, self.hidden_size)
        self.b_input_hidden = np.zeros((1, self.hidden_size))
        self.W_hidden_output = np.random.rand(self.hidden_size, self.output_size)
        self.b_hidden_output = np.zeros((1, self.output_size))
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    def d_sigmoid(self, x):
        return x * (1 - x)
    def train(self, input_data, target, epochs=1000, lr=0.2):
        for epoch in range(epochs):
            # Forward propagation
            hidden_layer_input = np.dot(input_data, self.W_input_hidden) + self.b_input_hidden
            hidden_layer_output = self.sigmoid(hidden_layer_input)
            output_layer_input = np.dot(hidden_layer_output, self.W_hidden_output) + 
                                                                                                                           self.b_hidden_output
            output = self.sigmoid(output_layer_input)
        
            # Backward propagation
            output_error = target - output
            output_grad = output_error * self.d_sigmoid(output)
            hidden_error = np.dot(output_grad, self.W_hidden_output.T)
            hidden_grad = hidden_error * self.d_sigmoid(hidden_layer_output)

            # Update weights and biases using gradient descent
            self.W_hidden_output += np.dot(hidden_layer_output.T, output_grad) * lr
            self.b_hidden_output += np.sum(output_grad, axis=0, keepdims=True) * lr
            self.W_input_hidden += np.dot(input_data.T, hidden_grad) * lr
            self.b_input_hidden += np.sum(hidden_grad, axis=0, keepdims=True) * lr
            
            # Optionally, print error every 1000 epochs
            if epoch % 1000 == 0:
                error = np.mean(np.square(target - output))  # Mean Squared Error
                print(f'Epoch {epoch}, Error: {error}')

# Example usage:
if __name__ == "__main__":
    # XOR problem: 4 samples, 2 input features, 1 output
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])
    
    # Create MLP with 2 input nodes, 4 hidden nodes, and 1 output node
    mlp = MLP(input_size=2, hidden_size=4, output_size=1)
    
    mlp.train(X, y, epochs=10000)		    # Train the model
   
    # Test the model after training
    print("Predictions after training:")
    hidden_layer_input = np.dot(X, mlp.W_input_hidden) + mlp.b_input_hidden
    hidden_layer_output = mlp.sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, mlp.W_hidden_output) + mlp.b_hidden_output
    predictions = mlp.sigmoid(output_layer_input)   
                print(predictions)

