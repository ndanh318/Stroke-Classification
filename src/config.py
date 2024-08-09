from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# models
models = {
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "decision_tree": DecisionTreeClassifier(),
    "k-nearest_neighbors": KNeighborsClassifier(),
}

# params
lr_params = {
    "preprocessor__num_features__imputer__strategy": ["mean", "median", "most_frequent"],
    "smt__sampling_strategy": ["minority", "not minority", "not majority", {1: 1000}, {1: 2000}, {1: 4000}],
    "model__penalty": ["l1", "l2"],
    "model__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    "model__solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"]
}
rf_params = {
    "preprocessor__num_features__imputer__strategy": ["mean", "median", "most_frequent"],
    "smt__sampling_strategy": ["minority", "not minority", "not majority", {1: 1000}, {1: 2000}, {1: 4000}],
    'model__n_estimators': [100, 200],
    'model__max_depth': [10, 20, 100, None],
    'model__max_features': ['auto', 'sqrt', None],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4, 10],
    'model__bootstrap': [True, False],
    'model__criterion': ['gini', 'entropy']
}
dtree_params = {
    "preprocessor__num_features__imputer__strategy": ["mean", "median", "most_frequent"],
    "smt__sampling_strategy": ["minority", "not minority", "not majority", {1: 1000}, {1: 2000}, {1: 4000}],
    'model__criterion': ['gini', 'entropy'],
    'model__splitter': ['best', 'random'],
    'model__max_depth': [1, 5, 10, 15, 20],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 4, 10],
    'model__max_features': ['auto', 'sqrt', 'log2', None]
}
knn_params = {
    "preprocessor__num_features__imputer__strategy": ["mean", "median", "most_frequent"],
    "smt__sampling_strategy": ["minority", "not minority", "not majority", {1: 1000}, {1: 2000}, {1: 4000}],
    'model__n_neighbors': [1, 5, 10, 15],
    'model__weights': ['uniform', 'distance'],
    'model__leaf_size': [1, 3, 5],
    'model__algorithm': ['auto', 'kd_tree']
}
params = [lr_params, rf_params, dtree_params, knn_params]
