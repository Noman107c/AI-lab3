import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the dataset
file_path = '/mnt/data/Book1.xlsx'
data = pd.read_excel(file_path)

# Identify categorical columns
categorical_cols = ['Gender']

# Encode categorical columns
le = LabelEncoder()
for col in categorical_cols:
    data[col] = le.fit_transform(data[col])

# Encode target variable 'Workout_Type'
data['Workout_Type'] = le.fit_transform(data['Workout_Type'])

# Separate features and target
X = data.drop(columns=['Workout_Type'])
y = data['Workout_Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)

# Display results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_mat)

# Decode predictions back to workout types
decoded_predictions = le.inverse_transform(y_pred)
print("Decoded Predictions:\n", decoded_predictions)
