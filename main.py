import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# 1. Loading CIFAR-10 data
print("🔍 Loading CIFAR-10...")
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train, y_test = y_train.flatten(), y_test.flatten()

# Critical Testing: Verify data shapes and types with graphs
print(f"Data shapes - x_train: {x_train.shape}, y_train: {y_train.shape}, x_test: {x_test.shape}, y_test: {y_test.shape}")
print(f"Data types - x_train: {x_train.dtype}, y_train: {y_train.dtype}, x_test: {x_test.dtype}, y_test: {y_test.dtype}")


# 2. Building the model
print("🏗️ Creating model...")
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Critical Testing: Print model summary to confirm architecture
model.summary()

# Plot model architecture to visually confirm layers 
try:
    tf.keras.utils.plot_model(model, to_file='model_architecture.png', show_shapes=True)
    from PIL import Image
    img = Image.open('model_architecture.png')
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.axis('off')
    plt.title("Model Architecture")
    plt.show()
except Exception as e:
    print(f"Could not plot model architecture: {e}")

# 3. Training
print("🚀 Training...")

# Add callback to monitor training progress
class TrainingLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"Epoch {epoch+1}: loss={logs['loss']:.4f}, accuracy={logs['accuracy']:.4f}, val_loss={logs['val_loss']:.4f}, val_accuracy={logs['val_accuracy']:.4f}")

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test), callbacks=[TrainingLogger()])

# 4. Predictions
print("Making predictions...")
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
pred_probs = prob_model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)

# Critical Testing: Verify prediction shapes and types
print(f"Prediction probabilities shape: {pred_probs.shape}, dtype: {pred_probs.dtype}")
print(f"Predicted labels shape: {pred_labels.shape}, dtype: {pred_labels.dtype}")

# 5. Visualizing 10 sample predictions
import random

print("\n Prediction visualization:")
plt.figure(figsize=(15, 6))
random_indices = random.sample(range(len(x_test)), 10)
for i, idx in enumerate(random_indices):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[idx])
    true = class_names[y_test[idx]]
    pred = class_names[pred_labels[idx]]
    conf = 100 * np.max(pred_probs[idx])
    color = 'green' if true == pred else 'red'
    plt.title(f"{pred} ({conf:.1f}%)\nTrue: {true}", color=color, fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()

# 6. Per-class accuracy
print("\n Per-class accuracy:")
conf_matrix = confusion_matrix(y_test, pred_labels)
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, acc in enumerate(class_accuracy):
    print(f"{class_names[i]}: {acc:.2%}")

# Critical Testing: Assert no NaNs in accuracy
assert not np.isnan(class_accuracy).any(), "NaN values found in class accuracy"

# Plot per-class accuracy as bar chart
plt.figure(figsize=(10, 6))
plt.bar(class_names, class_accuracy, color='skyblue')
plt.title("Per-Class Accuracy")
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()

# 7. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# 8. Classification Report
print("\n Classification Report:")
report = classification_report(y_test, pred_labels, target_names=class_names)
print(report)

# Critical Testing: Check classification report is not empty
assert len(report) > 0, "Classification report is empty"

# 9. Training History: Accuracy & Loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Train Acc')
plt.plot(epochs, val_acc, 'g', label='Val Acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 10. Confidence Histogram
confidences = np.max(pred_probs, axis=1)
plt.hist(confidences, bins=20, color='purple', alpha=0.7)
plt.title("Model Confidence on Test Set")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.grid(True)
plt.show()
