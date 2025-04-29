import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values
y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)
# Build the model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes for digits 0-9
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Train the model
model.fit(x_train, y_train_cat, epochs=5, batch_size=32, validation_split=0.2)
# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test_cat)
print(f'Test accuracy: {test_acc:.2f}')
# Pick a random sample from the test set
index = np.random.randint(0, len(x_test))
sample_image = x_test[index]
sample_label = y_test[index]
# Predict the digit
prediction = model.predict(sample_image.reshape(1, 28, 28))
predicted_class = np.argmax(prediction)
# Show the image and prediction
plt.imshow(sample_image, cmap='gray')
plt.title(f"Predicted: {predicted_class}, Actual: {sample_label}")
plt.axis('off')
plt.show()

