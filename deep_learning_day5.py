import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.fashion_mnist.load_data()
)

print("Training images shape:", x_train.shape)
print("Training labels shape:", y_train.shape)
print("Testing images shape:", x_test.shape)
print("Testing labels shape:", y_test.shape)



x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0


print("Minimum pixel value:", x_train.min())
print("Maximum pixel value:", x_train.max())


plt.imshow(x_train[0], cmap="gray", vmin=0, vmax=1)
plt.title(f"Label: {y_train[0]}")
plt.axis("off")
plt.show()


model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully!")

history = model.fit(
    x_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1
)

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print("\n--- Test Results ---")
print("Test loss:", round(test_loss, 4))
print("Test accuracy:", round(test_accuracy, 4))
print("Test accuracy (%):", round(test_accuracy * 100, 2))




# Class names for the 10 Fashion-MNIST categories
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Select one image from the test dataset
index = 0

image = x_test[index]
true_label = y_test[index]

# Ask the model to predict the image's class
probabilities = model.predict(
    image[np.newaxis, ...],
    verbose=0
)[0]

# Get the class with the highest probability
predicted_label = int(np.argmax(probabilities))

# Display the results
print("Actual label:", class_names[true_label])
print("Predicted label:", class_names[predicted_label])
print("Prediction score:", round(float(probabilities[predicted_label]), 4))

# Display the image
plt.imshow(image, cmap="gray", vmin=0, vmax=1)
plt.title(f"Predicted: {class_names[predicted_label]}")
plt.axis("off")
plt.show()


# Plot training and validation accuracy
plt.plot(history.history["accuracy"], label="Training accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid(True)
plt.show()



from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict the classes for all test images
test_probabilities = model.predict(x_test, verbose=0)
y_pred = np.argmax(test_probabilities, axis=1)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Display it
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

display.plot(xticks_rotation=45, cmap="Blues")
plt.title("Fashion-MNIST Confusion Matrix")
plt.tight_layout()
plt.show()


from sklearn.metrics import classification_report

print(
    classification_report(
        y_test,
        y_pred,
        target_names=class_names,
        digits=3
    )
)



# Save the trained model
model.save("fashion_mnist_model.keras")

print("Model saved successfully!")



import tensorflow as tf

# Load the previously saved model
loaded_model = tf.keras.models.load_model(
    "fashion_mnist_model.keras"
)

print("Model loaded successfully!")



# Predict one test image using the loaded model
probabilities = loaded_model.predict(
    x_test[0][np.newaxis, ...],
    verbose=0
)

predicted_label = int(np.argmax(probabilities[0]))

print("Predicted class:", class_names[predicted_label])

