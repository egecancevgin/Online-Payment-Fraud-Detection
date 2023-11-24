from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score
import pandas as pd

# Load the data
df = pd.read_csv('online_payments.csv')

# Drop rows with NaN values in the target variable
df.dropna(subset=['isFraud'], inplace=True)

# Separate majority and minority classes
df_majority = df[df['isFraud'] == 0]
df_minority = df[df['isFraud'] == 1]

# Downsample majority class
df_majority_downsampled = resample(df_majority,
                                   replace=False, # sample without replacement
                                   n_samples=len(df_minority), # match minority n
                                   random_state=42) # reproducible results

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True, # sample with replacement
                                 n_samples=len(df_majority_downsampled), # match majority n
                                 random_state=42) # reproducible results

# Combine minority class with downsampled majority class
df_resampled = pd.concat([df_majority_downsampled, df_minority_upsampled])

# Define the categorical and numerical features
cat_features = ['type']
num_features = ['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_features),
        ('cat', OneHotEncoder(), cat_features),
        ('imputer', SimpleImputer(strategy='mean'), num_features)])

# Define the pipeline
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])

# Define the hyperparameters to tune
param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_features': ['sqrt', 'log2'],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df_resampled.drop('isFraud', axis=1), 
                                                    df_resampled['isFraud'], 
                                                    test_size=0.2, 
                                                    random_state=42)

# Perform a grid search with cross-validation
grid_search = GridSearchCV(rf_pipeline, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best hyperparameters and mean cross-validation score
print("Best parameters:", grid_search.best_params_)
print("Mean accuracy:", grid_search.best_score_)

# Evaluate the model on the test set
y_pred = grid_search.predict(X_test)
test_accuracy = grid_search.score(X_test, y_test)
test_precision = precision_score(y_test, y_pred)
test_recall = recall_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)

print("Test accuracy:", test_accuracy)
print("Test precision:", test_precision)
print("Test recall:", test_recall)
print("Test F1 score:", test_f1)
