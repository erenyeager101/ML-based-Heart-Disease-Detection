# import pickle
# from sklearn.ensemble import RandomForestClassifier
# import pandas as pd
#
# # Assuming 'dataset' contains your data
# dataset = pd.read_csv("heart.csv")
# predictors = dataset.drop("target", axis=1)
# target = dataset["target"]
#
# # Train your Random Forest model
# rf = RandomForestClassifier(random_state=42)
# rf.fit(predictors, target)
#
# # Save the trained model to a file
# with open('model_randomforestversion2', 'wb') as f:
#     pickle.dump(rf, f)


import joblib

from pythonProject.model import rf

# Save the trained model to a file
joblib.dump(rf, 'model_randomforestversion2.joblib')

import joblib

# Load the trained model
model = joblib.load('model_randomforestversion2.joblib')

