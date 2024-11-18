import numpy as np
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

age = ['youth', 'youth', 'middle_aged', 'senior', 'senior', 'senior', 'middle_aged', 'youth', 'youth', 
       'senior', 'youth', 'middle_aged', 'middle_aged', 'senior']
income = ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 
          'medium', 'medium', 'high', 'medium']
student = ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no']
credit_rating = ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 
                 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent']
decision = ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']

# Encode categorical variables
age_encoded = le.fit_transform(age)
income_encoded = le.fit_transform(income)
student_encoded = le.fit_transform(student)
credit_rating_encoded = le.fit_transform(credit_rating)
decision_encoded = le.fit_transform(decision)

# Combine features
features = list(zip(age_encoded, income_encoded, student_encoded, credit_rating_encoded))

# Train-Test Split (use entire set for small dataset)
features_train = features
label_train = decision_encoded

# Train Naïve Bayes model
model = GaussianNB()
model.fit(features_train, label_train)

# Predict for youth/medium/yes/fair
input_instance = [[le.transform(['youth'])[0], le.transform(['medium'])[0], le.transform(['yes'])[0], le.transform(['fair'])[0]]]
prediction = model.predict(input_instance)
print(f"Prediction for youth/medium/yes/fair: {le.inverse_transform(prediction)}")

# Evaluate performance
predicted = model.predict(features_train)
conf_mat = confusion_matrix(label_train, predicted)
print(f"Confusion Matrix:\n{conf_mat}")
accuracy = accuracy_score(label_train, predicted)
print(f"Accuracy: {accuracy}")
