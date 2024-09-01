import pygame
import random
import numpy as np

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((600, 400))
pygame.display.set_caption("Neural Network Guessing Game")

# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Simple Neural Network class with training
class SimpleNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        # Initialize weights with random values
        self.weights1 = np.random.randn(input_size, hidden_size)
        self.weights2 = np.random.randn(hidden_size, output_size)
        self.learning_rate = learning_rate

    def feedforward(self, x):
        # Forward pass
        self.input = x.reshape(1, -1)  # Reshape to a 2D array
        self.hidden = sigmoid(np.dot(self.input, self.weights1))
        self.output = sigmoid(np.dot(self.hidden, self.weights2))
        return self.output

    def backpropagate(self, y_true):
        # Calculate error
        output_error = y_true - self.output
        output_delta = output_error * sigmoid_derivative(self.output)

        hidden_error = output_delta.dot(self.weights2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # Update weights (notice the correct use of dot products)
        self.weights2 += self.hidden.T.dot(output_delta) * self.learning_rate
        self.weights1 += self.input.T.dot(hidden_delta) * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            for i in range(len(X)):
                # Forward pass
                self.feedforward(X[i])
                # Backward pass and weight update
                self.backpropagate(y[i])

# Function to generate training data
def generate_training_data(num_samples=100):
    X = []
    y = []
    for _ in range(num_samples):
        shape_type = random.choice(["circle", "square"])
        size = random.randint(20, 100)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # Features: size normalized and average color intensity normalized
        features = [size / 100, sum(color) / (3 * 255)]
        X.append(features)
        
        # Labels: circle=1, square=0
        label = 1 if shape_type == "circle" else 0
        y.append([label])
    return np.array(X), np.array(y)

# Function to generate a random shape for the game
def generate_shape():
    shape_type = random.choice(["circle", "square"])
    size = random.randint(20, 100)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return shape_type, size, color

# Function to draw the shape on the screen
def draw_shape(shape_type, size, color):
    if shape_type == "circle":
        pygame.draw.circle(screen, color, (300, 200), size)
    elif shape_type == "square":
        pygame.draw.rect(screen, color, (300 - size, 200 - size, size * 2, size * 2))

# Initialize the Neural Network
nn = SimpleNN(input_size=2, hidden_size=3, output_size=1, learning_rate=0.1)

# Generate and prepare training data
X_train, y_train = generate_training_data(num_samples=500)

# Train the Neural Network
print("Training the Neural Network...")
nn.train(X_train, y_train, epochs=10000)
print("Training completed!")

# Main game loop
running = True
font = pygame.font.Font(None, 36)

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Generate a new random shape
    shape_type, size, color = generate_shape()

    # Fill screen with white background
    screen.fill((255, 255, 255))

    # Draw the shape
    draw_shape(shape_type, size, color)

    # Prepare the features for the Neural Network
    features = np.array([size / 100, sum(color) / (3 * 255)])  # Normalize the inputs

    # Neural Network makes a guess
    guess = nn.feedforward(features)
    predicted_shape = "circle" if guess > 0.5 else "square"

    # Display the Neural Network's guess
    text = font.render(f"Neural Network guessed: {predicted_shape}", True, (0, 0, 0))
    screen.blit(text, (150, 350))

    # Optionally, display the actual shape type for comparison
    actual_text = font.render(f"Actual shape: {shape_type}", True, (0, 0, 0))
    screen.blit(actual_text, (150, 300))

    # Update the display
    pygame.display.flip()

    # Delay to see the result
    pygame.time.delay(2000)

# Quit Pygame
pygame.quit()
