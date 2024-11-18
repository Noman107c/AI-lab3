import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# Sample dataset for Task 1 (based on your Fig 1)
weather = ['Sunny', 'Overcast', 'Rainy', 'Sunny', 'Sunny', 'Overcast', 'Rainy']
temperature = ['Hot', 'Mild', 'Mild', 'Cool', 'Hot', 'Mild', 'Cool']
play = ['No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes']

# Label Encoding
le = preprocessing.LabelEncoder()
weather_encoded = le.fit_transform(weather)
temperature_encoded = le.fit_transform(temperature)
play_encoded = le.fit_transform(play)

# Combine features
features = list(zip(weather_encoded, temperature_encoded))

# Train-Test Split (using full dataset since it's small)
features_train = features
label_train = play_encoded

# Train Naïve Bayes model
model = GaussianNB()
model.fit(features_train, label_train)

# Predict for "Overcast" (encoded as 0) and "Mild" (encoded as 1)
prediction = model.predict([[le.transform(['Overcast'])[0], le.transform(['Mild'])[0]]])
print(f"Prediction for Overcast, Mild: {le.inverse_transform(prediction)}")

# Confusion Matrix (if needed)
predicted = model.predict(features_train)
conf_mat = confusion_matrix(label_train, predicted)
print(f"Confusion Matrix:\n{conf_mat}")
accuracy = accuracy_score(label_train, predicted)
print(f"Accuracy: {accuracy}")
