"""Nested cross-validation with model selection (companion to the figure).

Step numbers in the comments refer to the numbered steps in
figures/nested_cv_workflow.svg. The inner optimizer here is a randomized
search; replace it with TPE (e.g. Optuna, hyperopt) or any other
hyperparameter optimization method without changing the structure.
"""

from scipy.stats import loguniform, randint
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = load_breast_cancer(return_X_y=True)

# Step 1: hold out a test set that is used exactly once, at the very end.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=0
)

# Step 2: the models to compare, each with its hyperparameter search space.
# Scaling lives inside the pipeline, so it is fitted on each fold's train part only.
candidates = {
    "KNN": (
        make_pipeline(StandardScaler(), KNeighborsClassifier()),
        {"kneighborsclassifier__n_neighbors": randint(1, 30)},
    ),
    "SVM": (
        make_pipeline(StandardScaler(), SVC()),
        {
            "svc__C": loguniform(1e-2, 1e2),
            "svc__gamma": loguniform(1e-4, 1e0),
        },
    ),
}

# Step 3: outer folds, identical for every model.
outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)


def make_search(estimator, space):
    # Steps 4-6: inner k-fold CV + optimizer (random search here, so it does not
    # use earlier losses; TPE would). The best hyperparameters are
    # refitted on the whole outer train part (RandomizedSearchCV refit=True).
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=1)
    return RandomizedSearchCV(
        estimator, space, n_iter=15, cv=inner_cv, scoring="accuracy", random_state=0
    )


# Steps 7-8: for every outer fold, tune on its train part, score on its held-out
# part, then average the three outer scores per model.
results = {}
for name, (estimator, space) in candidates.items():
    scores = cross_val_score(
        make_search(estimator, space), X_train, y_train, cv=outer_cv, scoring="accuracy"
    )
    results[name] = (scores.mean(), scores.std())
    print(f"{name}: outer score = {scores.mean():.3f} +/- {scores.std():.3f}")

# Step 9: pick the best model. These outer scores are for selection only.
best_name = max(results, key=lambda name: results[name][0])
print(f"Selected model: {best_name}")

# Step 10: tune the winner once more on the full training set with ONE plain
# k-fold CV (not nested), then test once on the test set.
estimator, space = candidates[best_name]
final_search = make_search(estimator, space).fit(X_train, y_train)
print(f"Best hyperparameters: {final_search.best_params_}")
print(f"Final test score: {final_search.score(X_test, y_test):.3f}")
