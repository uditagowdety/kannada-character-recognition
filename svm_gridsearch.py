import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
import ast
import joblib

# Read the CSV file into a DataFrame
df = pd.read_csv("hog_feature_vectors.csv")

print("loaded dataframe...")

# Convert string representations of lists into actual lists
df['feature_vector'] = df['feature_vector'].apply(lambda x: np.array(ast.literal_eval(x)))

# Stack all the feature vectors into a 2D array (shape: [num_samples, num_features])
features = np.stack(df['feature_vector'].values)

# Extract the labels
labels = df['class']

print("built feature and label vectors...")

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],                    # Regularization parameter
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],   # Different kernels to try
    'decision_function_shape': ['ovr', 'ovo']  # One-vs-Rest or One-vs-One
}

# Initialize the SVM classifier
classifier = svm.SVC()

# Initialize GridSearchCV with the classifier and parameter grid
grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)

print("starting grid search now...")

# Fit the model using GridSearchCV
grid_search.fit(X_train, y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_classifier = grid_search.best_estimator_

print("Best parameters found:", best_params)

# Predict on the test set using the best model
y_pred = best_classifier.predict(X_test)

# Evaluate the classifier
print("classification report:")
print(classification_report(y_test, y_pred))

print("accuracy score:", accuracy_score(y_test, y_pred))

# If you want to see the cross-validation scores for the best model
print("mean cross-validation score:", grid_search.best_score_)

joblib.dump(best_classifier,"hog_model.pkl")
