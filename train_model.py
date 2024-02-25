import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# Load the Iris dataset for demonstration
iris = load_iris()
X, y = iris.data, iris.target

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X, y)

# Save the trained model to a file using pickle
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)