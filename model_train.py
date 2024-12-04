# sorting the data into training and test sets and fitting into the model

import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
#from sklearn.metrics import classification_report, accuracy_score

# loading the dataset
df = pd.read_csv('asl_features.csv')

# selecting the labels and features
X = df.drop(columns=['filename', 'label']).values
y = df['label'].values

# defining the model
svc = SVC(kernel='rbf', C=1.0, gamma='scale')

# dividing the dataset into train and test sets
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.3, random_state=42)

# print(f"Shape of X_train: {X_train.shape}")
# print(f"Shape of X_test: {X_test.shape}")
# print(f"Shape of y_train: {y_train.shape}")
# print(f"Shape of y_test: {y_test.shape}")

# fitting the data into the model
svc.fit(X_train, y_train)

# evaluating the model
y_pred = svc.predict(X_test)
# print(classification_report(y_test, y_pred))
# print('accuracy: ', accuracy_score(y_test, y_pred))

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly']
}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
print(grid.best_params_)

# save model to a file
joblib.dump(svc, 'svc_model.pkl')
print('model successfully saved!')

