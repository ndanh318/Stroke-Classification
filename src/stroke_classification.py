import os
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import seaborn as sns
from config import *

from ydata_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc


def classifier(cls, params, title):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Randomize Grid Search
    n_iter_search = 30
    model = RandomizedSearchCV(cls,
                               param_distributions=params,
                               n_iter=n_iter_search,
                               random_state=42,
                               scoring="recall",
                               cv=cv,
                               n_jobs=6,
                               verbose=2)
    model.fit(x_train, y_train)
    print("The best score is {}".format(model.best_score_))
    print("The best params are {}".format(model.best_params_))

    # save model
    model_path = "../models"
    model_pkl_file = "{}_model.pkl".format(title)
    with open(os.path.join(model_path, model_pkl_file), 'wb') as file:
        pickle.dump(model, file)

    # evaluate
    y_predict = model.best_estimator_.predict(x_test)
    y_prob = model.best_estimator_.predict_proba(x_test)[:, 1]
    print('Accuracy score: {:.2f}'.format(accuracy_score(y_test, y_predict)))
    print('Precision score: {:.2f}'.format(precision_score(y_test, y_predict)))
    print('Recall score: {:.2f}'.format(recall_score(y_test, y_predict)))

    # Visualize
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)

    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_predict)
    sns.heatmap(cm, annot=True, cbar=False, fmt="d", linewidths=.5, cmap="Blues", ax=ax1)
    ax1.set_title("Confusion Matrix")
    ax1.set_xlabel("Predicted class")
    ax1.set_ylabel("Actual class")
    fig.tight_layout()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax2.plot(fpr, tpr, lw=2, label='AUC: {:.2f}'.format(auc(fpr, tpr)))
    ax2.plot([0, 1], [0, 1],
             linestyle='--',
             color=(0.6, 0.6, 0.6),
             label='Random guessing')
    ax2.plot([0, 0, 1], [0, 1, 1],
             linestyle=':',
             color='black',
             label='Perfect performance')
    ax2.set_xlim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlabel('False Positive Rate (FPR)')
    ax2.set_ylabel('True Positive Rate (TPR)')
    ax2.set_title('Receiver Operator Characteristic (ROC) Curve')
    ax2.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("../images/{}".format(title), bbox_inches="tight")
    plt.show()


if __name__ == '__main__':
    data = pd.read_csv("../data/stroke_classification.csv")
    # profile = ProfileReport(data, title="Stroke Report", explorative=True)
    # profile.to_file("stroke_report.html")
    data = data.drop("pat_id", axis=1)
    print(data.info())

    # correlation
    numeric = [data.columns[index] for index, dtype in enumerate(data.dtypes) if dtype != "object"]
    plt.figure(figsize=(15, 10))
    corr = data[numeric].corr()
    sns.heatmap(corr, annot=True, fmt=".1f")
    plt.savefig("../images/correlation", bbox_inches="tight")
    plt.show()

    # split data
    target = "stroke"
    x = data.drop(target, axis=1)
    y = data[target]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # preprocessing
    numeric = [x_train.columns[index] for index, dtype in enumerate(x_train.dtypes) if dtype != "object"]
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("one_hot", OneHotEncoder(handle_unknown="ignore"), ["gender"]),
        ("num_features", num_transformer, numeric)
    ])

    # classifier
    for param, (name, model) in zip(params, models.items()):
        cls = ImbPipeline(steps=[
            ("preprocessor", preprocessor),
            ("smt", SMOTE(random_state=42)),
            ("model", model)
        ])
        classifier(cls, param, name)