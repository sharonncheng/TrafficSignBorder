# evaluate_model.py
import numpy as np
from tensorflow.keras.models import load_model
from data_prep import X_test, y_test
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the model
model = load_model('models/best_model.h5')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

# additional analysis: confusion matrix

# Predict the values from the test dataset
Y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors 
Y_pred_classes = np.argmax(Y_pred, axis=1) 
# Convert test observations to one hot vectors
Y_true = np.argmax(y_test, axis=1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 

# visualization

plt.figure(figsize=(10,8))
sns.heatmap(confusion_mtx, annot=True, fmt="d")
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()
