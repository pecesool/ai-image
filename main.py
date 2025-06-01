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

# 3. Training
print("🚀 Training...")
history = model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 4. Predictions
print("🔮 Making predictions...")
prob_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
pred_probs = prob_model.predict(x_test)
pred_labels = np.argmax(pred_probs, axis=1)

# 5. Per-class accuracy
print("\n📊 Per-class accuracy:")
conf_matrix = confusion_matrix(y_test, pred_labels)
class_accuracy = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
for i, acc in enumerate(class_accuracy):
    print(f"{class_names[i]}: {acc:.2%}")

# 6. Visualizing 10 sample predictions
print("\n🖼️ Prediction visualization:")
plt.figure(figsize=(15, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i])
    true = class_names[y_test[i]]
    pred = class_names[pred_labels[i]]
    conf = 100 * np.max(pred_probs[i])
    color = 'green' if true == pred else 'red'
    plt.title(f"{pred} ({conf:.1f}%)\nTrue: {true}", color=color, fontsize=9)
    plt.axis('off')
plt.tight_layout()
plt.show()

# 7. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("🧮 Confusion Matrix")
plt.show()

# 8. Classification Report
print("\n📋 Classification Report:")
print(classification_report(y_test, pred_labels, target_names=class_names))

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
plt.title('📈 Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Train Loss')
plt.plot(epochs, val_loss, 'g', label='Val Loss')
plt.title('📉 Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 10. Confidence Histogram
confidences = np.max(pred_probs, axis=1)
plt.hist(confidences, bins=20, color='purple', alpha=0.7)
plt.title("🔮 Model Confidence on Test Set")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.grid(True)
plt.show()