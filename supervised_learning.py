import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.preprocessing import power_transform, FunctionTransformer, PowerTransformer, StandardScaler, RobustScaler, QuantileTransformer, Normalizer, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.kernel_approximation import Nystroem
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


def split_the_data(df, column, cv, stratify_bool):
    X = df.drop(columns=column, axis=1)
    y = df[column]

    baseline_experiment(X, y, cv)

    # Stratify makes it so that the proportion of values in the sample in our test group will be the same as the
    # proportion of values provided to parameter stratify
    if stratify_bool:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42, stratify=y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size=0.20, random_state=42)

    return X, y, X_train, X_test, y_train, y_test


def validation_curve_scores(estimator, X, y, param, param_possibilites, score='accuracy', cv=None, n_jobs=None):

    train_scores, test_scores = validation_curve(estimator, X, y, cv=cv, param_name=param, param_range=param_possibilites, scoring=score, n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    return train_scores_mean, train_scores_std, test_scores_mean, test_scores_std


def plot_curves(estimator, title, X, y, score, param, param_options, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    if axes is None:
        _, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes, return_times=True, scoring=score, random_state=42)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="k",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    # Plot validation curve
    print("================")
    print(estimator)
    print(param)
    print(param_options)
    print("================")
    train_scores_mean, train_scores_std, test_scores_mean, test_scores_std = validation_curve_scores(
        estimator, X, y, param, param_options, score, cv, n_jobs)
    axes[3].grid()
    axes[3].fill_between(param_options, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="darkorange")
    axes[3].fill_between(param_options, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="navy")
    axes[3].plot(param_options, train_scores_mean, 'o-',
                 label="Training score", color="darkorange")
    axes[3].plot(param_options, test_scores_mean, 'o-',
                 label="Cross-validation score", color="navy")
    axes[3].legend(loc="best")
    axes[3].set_title(title.replace("Learning", "Validation"))
    axes[3].set_xlabel(param)
    axes[3].set_ylabel("Score")

    return plt


def baseline_experiment(x_data, y_data, cross_val):
    model = DummyClassifier(strategy='prior', random_state=0)
    # Create a title for each column and the console by using str() and
    # slicing away useless parts of the string
    model_title = str(type(model)).split(".")[-1][:-2][:-len("Classifier")]
    scores = cross_val_score(model, x_data, y_data, cv=cross_val, scoring='accuracy')
    print("-----------")
    print("Baseline Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    scores = cross_val_score(model, x_data, y_data, cv=cross_val, scoring='f1_macro')
    print("Baseline F1 Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("-----------")
    model.fit(x_data, y_data)
    print(classification_report(y_data, model.predict(x_data), output_dict=True))


def load_and_describe_data(file):
    data = pd.read_csv(file)

    if file == 'data/output/heart-disease-data.csv':
        data['ca'] = pd.to_numeric(data['ca'], errors='coerce').interpolate()
        data['thal'] = pd.to_numeric(data['thal'], errors='coerce').interpolate()

    assert not data.isnull().values.any()

    # Frequency Statistics - run once
    # csv_file = file.split('/')
    #
    # describe_file = 'data/statistics/description-' + csv_file[2]
    # data.describe(include='all').to_csv(describe_file, header=True)
    # # Less than or greater than -1/+1 is highly skewed
    # skew_file = 'data/statistics/skew-' + csv_file[2]
    # data.skew(axis=0).to_csv(skew_file, header=True)
    #
    # plot_data(csv_file[2], data)

    return data


def plot_data(file, df):
    plt.figure()
    df.plot.box()
    ax = plt.axes()
    ax.set_xticklabels(labels=df.columns, rotation='vertical', fontsize=6)
    plot_file = file[:-4] + ".png"
    plt.savefig('figures/' + plot_file, dpi=150)


def print_statistics(df, target):
    print(df.describe(include='all'))
    print(df.skew(axis=0))
    print(df[target].value_counts())


def set_parameters(title, max_columns):
    param = []

    if title == 'DecisionTreeClassifier':
        param = [
            {
                'clf__criterion': ['gini'],
                'clf__max_features': range(1, max_columns+1, 2),
                'clf__max_depth': range(1, max_columns+1, 2),
                # 'clf__min_samples_split': range(10, 1000, 100),
                'clf__min_samples_leaf': range(1, 52, 10)
            }
        ]
    elif title == 'MLPClassifier':
        param = [
            {
                'clf__solver': ['adam'],
                # [(50,), (100,), (200,), (50, 2), (100, 2), (200, 2)],
                'clf__hidden_layer_sizes': [(50,), (100,), (50, 2), (100, 2)],
                'clf__alpha': [1e-5],
                'clf__max_iter': [9000]
            }
        ]
    elif title == 'AdaBoostClassifier':
        param = [
            {
                'clf__n_estimators': [25, 75, 100, 150],
                'clf__learning_rate': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
            }
        ]
    elif title == 'RandomForestClassifier':
        param = [
            {
                'clf__n_estimators': [750, 1000, 1250],
                'clf__max_depth': range(1, max_columns+1, 3),
                'clf__min_samples_leaf': range(11, 41, 10)
            }
        ]
    elif title == 'LinearSVC':
        param = [
            {
                'clf__penalty': ['l1', 'l2'],
                'clf__dual': [False],
                'clf__fit_intercept': [False],
                'clf__tol': [1e-03, 1e-04, 1e-05],
                'clf__max_iter': [100000]
            }
        ]
    elif title == 'KNeighborsClassifier':
        param = [
            {
                'clf__n_neighbors': range(30, 71, 2),
                'clf__weights': ['uniform', 'distance']
            }
        ]
    elif title == 'SVC':
        param = [
            {
                'clf__gamma': ['scale', 'auto'],
                'clf__decision_function_shape': ['ovo']
            }
        ]

    return param


def experiment_inputs():
    # Comment the code below accordingly to run for specific files
    inputs = {
        'experiment': [
            'Credit Card',
            # 'Poker Hand',
            # 'Heart Disease'
        ],
        'data_sets_filenames': [
            'data/output/credit-card-data.csv',
            # 'data/output/sampled-poker-hand-data.csv',
            # 'data/output/heart-disease-data.csv'
        ],
        'feature_label': [
            'default payment next month',
            # 'Poker Hand',
            # 'num'
        ],
        'num_cross_validation': [
            5,
            # 3,
            # 10
        ],
        'models': [
            # DecisionTreeClassifier(random_state=42),
            # DummyClassifier(strategy='prior', random_state=0),
            # MLPClassifier(random_state=42),
            # Uncomment for credit card
            # AdaBoostClassifier(DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=51)),
            # Uncomment for poker
            # AdaBoostClassifier(DecisionTreeClassifier(randoms_state=42, max_depth=11, min_samples_leaf=1)),
            # RandomForestClassifier(random_state=42, n_jobs=-1),
            # SVC(random_state=42),
            KNeighborsClassifier()
        ],
        'grid_search_bool': [
            # True,
            # True,
            # True,
            # True,
            # True,
            # True,
            True
        ],
        'steps': [
            # Decision Tree for Poker Hand no preprocessing needed
            # [
            #     ('transform', PowerTransformer()),
            #     ('scale', RobustScaler())
            # ],
            # Dummy Classifier
            # [
            # ],
            # Only preprocess on credit card data
            # MLP Classifier
            # [
            #     ('transform', PowerTransformer()),
            #     ('scale', RobustScaler()),
            # ],
            # # AdaBoost
            # [
            #     ('transform', PowerTransformer()),
            #     ('scale', RobustScaler()),
            # ],
            # Random Forest
            # [
            #     ('transform', PowerTransformer()),
            #     ('scale', RobustScaler())
            # ],
            # SVC
            # [
            #     # ('transform', PowerTransformer()),
            #     # ('scale', RobustScaler()),
            #     # ('scale', MinMaxScaler(feature_range=(-1, 1)))
            #     ('kernel_transform', Nystroem(gamma=.2, random_state=1))
            # ],
            # # KNN
            [
                ('transform', PowerTransformer()),
                ('scale', RobustScaler())
            ]
        ],
        'params_to_test': [
            [
                # {
                #     'clf__max_depth': range(1, 50, 2)
                # },
                # {
                #     'clf__alpha': np.linspace(1e-10, 1e-3, 25)
                # },
                # {
                #     'clf__n_estimators': np.linspace(10, 500, 25, dtype=int)
                # },
                # {
                #     'clf__min_samples_split': np.linspace(2, 500, 25, dtype=int)
                # },
                # {
                #     'clf__tol': np.linspace(1e-10, 1e-1, 25)
                # },
                {
                    'clf__n_neighbors': np.linspace(3, 53, 25, dtype=int)
                }
            ]
        ]
    }

    return inputs


def search_for_hyperparameters(model_title, params, steps, cv, X_train, y_train):
    pipe = Pipeline(steps)
    print("Starting the Grid Search for %s" % model_title)

    grid_clf = GridSearchCV(pipe, param_grid=params, scoring='accuracy', cv=cv, verbose=True, n_jobs=-1)
    grid_clf.fit(X_train, y_train)

    return grid_clf


# More control to repeat what previous model we should run the learning curve experiment on
def get_previous_model(model, steps, experiment):
    print("Model: " + str(model))
    clf = None
    if experiment == "Credit Card":
        if model == "DecisionTreeClassifier":
            # With Feature transform and Robust scaling
            clf = DecisionTreeClassifier(max_depth=9, max_features=19, min_samples_split=910)
        elif model == "MLPClassifier":
            # With Feature Transform and Robust Scaling
            clf = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-05, max_iter=9000, solver='adam')
        elif model == "AdaBoostClassifier":
            clf = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.7, n_estimators=100)
        elif model == "RandomForestClassifier":
            clf = RandomForestClassifier(max_depth=10, min_samples_leaf=21, n_estimators=1000)
        elif model == "KNeighborsClassifier":
            clf = KNeighborsClassifier(n_neighbors=65, weights='distance')
        else:
            clf = DummyClassifier(strategy='prior', random_state=0)

    elif experiment == "Poker Hand":
        # Dummy Classifier has
        # Baseline Accuracy: 0.63 (+/- 0.00)
        # Baseline F1 Accuracy: 0.08 (+/- 0.00)

        if model == "DecisionTreeClassifier":
            # With Feature transform and Robust scaling
            # ('transform', PowerTransformer()),
            # ('scale', RobustScaler())
            clf = DecisionTreeClassifier(criterion='gini', max_depth=77, max_features=7, min_samples_split=10,
                                         random_state=42)

        elif model == "MLPClassifier":
            # ('scale', RobustScaler()),
            # {'clf__alpha': 1e-05, 'clf__hidden_layer_sizes': (100, 2), 'clf__max_iter': 9000, 'clf__solver': 'adam'}
            clf = MLPClassifier(hidden_layer_sizes=(100, 2), alpha=1e-05, max_iter=9000, solver='adam',
                                random_state=42)

        elif model == "AdaBoostClassifier":
            # ('transform', PowerTransformer()),
            # ('scale', RobustScaler()),
            # {'clf__learning_rate': 0.9, 'clf__n_estimators': 500}

            clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=10, min_samples_leaf=1),
                                     learning_rate=0.9, n_estimators=500, random_state=42)

        elif model == "RandomForestClassifier":
            # ('transform', PowerTransformer()),
            # ('scale', RobustScaler())
            # {'clf__max_depth': 10, 'clf__min_samples_leaf': 1, 'clf__n_estimators': 1000}

            clf = RandomForestClassifier(max_depth=58, min_samples_leaf=1, n_estimators=100, random_state=42)

        elif model == "KNeighborsClassifier":
            # TODO
            # ('transform', PowerTransformer()),
            # ('scale', RobustScaler())
            # {'clf__n_neighbors': 21, 'clf__weights': 'distance'}
            # 'accuracy': 0.314082239890737, 'macro avg': {'precision': 0.3391855974907591,
            # 'recall': 0.4384785848505501, 'f1-score': 0.29325725336220687, 'support': 20501}

            clf = KNeighborsClassifier(n_neighbors=45, weights='distance')

        elif model == "LinearSVCClassifier":
            # ('kernel_transform', Nystroem(gamma=.2, random_state=1))
            # {'clf__dual': False, 'clf__fit_intercept': False, 'clf__max_iter': 100000, 'clf__penalty': 'l1',
            #  'clf__tol': 0.001}
            # {'accuracy': 0.2690600458514219, 'macro avg': {'precision': 0.12389730160404613,
            # 'recall': 0.13910247772057793, 'f1-score': 0.0796326760718585, 'support': 20501}
            clf = LinearSVC(dual=False, fit_intercept=False, max_iter=100000, penalty='l1', tol=0.001, random_state=42)

        else:
            clf = DummyClassifier(strategy='prior', random_state=42)

    elif experiment == "Heart Disease":
        if model == "DecisionTreeClassifier":
            # With Feature transform and Robust scaling
            clf = DecisionTreeClassifier(max_depth=19, max_features=9, min_samples_split=10)
        elif model == "MLPClassifier":
            # With Feature transform and Robust scaling
            clf = MLPClassifier(hidden_layer_sizes=(50,), alpha=1e-05, max_iter=9000, solver='adam')
        elif model == "AdaBoostClassifier":
            clf = AdaBoostClassifier(DecisionTreeClassifier(), learning_rate=0.6, n_estimators=50)
        elif model == "RandomForestClassifier":
            clf = RandomForestClassifier(max_depth=7, min_samples_leaf=11, n_estimators=10)
        else:
            clf = DummyClassifier(strategy='prior', random_state=0)

    new_steps = add_pipeline_step(steps, ('clf', clf))
    pipeline = Pipeline(new_steps)
    # print("========")
    # print("Previous Model: ")
    # print(pipeline)
    # print("========")
    return pipeline


def add_pipeline_step(pipeline_steps, new_tuple):
    new_steps = []
    for st in pipeline_steps:
        new_steps.append(st)
    new_steps.append(new_tuple)

    return new_steps


if __name__ == "__main__":
    inputs = experiment_inputs()

    # Upsample the poker hand data
    # Transform the credit card data
    for i, filename in enumerate(inputs['data_sets_filenames']):
        # Setup
        experiment = inputs['experiment'][i]
        cv = inputs['num_cross_validation'][i]
        bool_list = inputs['grid_search_bool']
        models = inputs['models']
        target_feature = inputs['feature_label'][i]
        testing_params_to_test = inputs['params_to_test'][i]

        scores = ['accuracy', 'f1_macro']

        # Classification problem so stratify
        stratify = True

        data = load_and_describe_data(filename)

        print('%s - %s - %s' % (filename, experiment, cv))

        trimmed_experiment = experiment.replace(" ", "")

        for j, model in enumerate(inputs['models']):
            # More Setup
            model_title = str(type(model)).split(".")[-1][:-2]
            step = inputs['steps'][j]
            search_the_grid = bool_list[j]
            for k, v in testing_params_to_test[j].items():
                param_to_test = k
                param_range = v

            # print_statistics(data, target_feature)

            X, y, X_train, X_test, y_train, y_test = split_the_data(data, target_feature, cv, stratify)
        
            print(y.value_counts())

            for score in scores:
                title = "%s Learning Curve - %s - on %s" % (model_title, experiment, score)
                workfile = "analysis/%s-%s-%s-parameters-analysis.txt" % (model_title, trimmed_experiment, score)

                f = open(workfile, 'w')
                f.write("Tuning hyper-parameters for %s\n" % score)
                f.write("%s\n" % title)

                print(title)
                # print(bool_list[j])

                max_columns = len(X.columns)
                # Unused max_samples to determine min_sample_split
                max_samples = int(X.shape[0] * 0.05)
                print("My Max Samples: " + str(max_samples))
                print("Entire Sample Size: " + str(X.shape[0]))
                print("Training Sample Size: " + str(X_train.shape[0]))
                print("Test Sample Size: " + str(X_test.shape[0]))

                tuned_parameters = set_parameters(model_title, max_columns)
                pipeline = None

                new_step = add_pipeline_step(step, ('clf', model))

                clf = search_for_hyperparameters(model_title, tuned_parameters, new_step, cv, X_train, y_train)
                f.write("Best parameters found on development set:\n")
                f.write(str(clf.best_params_))
                f.write("\n")
                f.write("Best Pipeline:\n")
                f.write(str(clf.best_estimator_))
                f.write("\n")

                y_true, y_pred = y_test, clf.predict(X_test)
                print(confusion_matrix(y_true, y_pred))
                print(classification_report(y_true, y_pred))

                f.write("Detailed classification report:\n")
                f.write("The model is trained on the full development set.\n")
                f.write("The scores are computed on the full evaluation set.\n")
                report = classification_report(y_true, y_pred, output_dict=True)
                f.write(str(report))
                f.write("\n")
                f.write(np.array2string(confusion_matrix(y_true, y_pred)))
                f.write("\n")

                # title = "%s Learning Curve - %s" % (model_title, experiment)
                # pipe = get_previous_model(model_title, step, experiment)
                # pipe.fit(X_train, y_train)
                # y_true, y_pred = y_test, pipe.predict(X_test)

                # print(X.shape)
                max_samples = X.shape[0] * 0.05
                # print(tuned_parameters[0])
                # print(clf.best_estimator_)

                plot_curves(clf.best_estimator_, title, X, y, score, param_to_test, param_range, cv=cv, n_jobs=-1)
                plt.savefig('figures/' + title + '.png', dpi=150)
                reportfile = "analysis/%s-%s-report.txt" % (model_title, trimmed_experiment)
                # f = open(reportfile, 'w')
                # f.write("Best parameters set found on development set:\n")
                # f.write(str(pipe))
                # f.write("\n")
                # f.write("Detailed classification report:\n")
                # f.write("The model is trained on the full development set.\n")
                # f.write("The scores are computed on the full evaluation set.\n")
                # report = classification_report(y_true, y_pred, output_dict=True)
                # f.write(str(report))
                # f.write("\n")
                # f.write(np.array2string(confusion_matrix(y_true, y_pred)))
                # f.write("\n")
                cross_scores = cross_val_score(clf.best_estimator_, X, y, cv=cv, scoring='accuracy')
                print("-----------")
                f.write("\n-----------\n")
                print(model_title)
                print("Cross-Val Accuracy: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
                f.write("Cross-Val Accuracy: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
                cross_scores = cross_val_score(clf.best_estimator_, X, y, cv=cv, scoring='f1_macro')
                print("Cross-Val F1 Accuracy: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
                f.write("Cross-Val F1 Accuracy: %0.2f (+/- %0.2f)" % (cross_scores.mean(), cross_scores.std() * 2))
                print("-----------")
                f.write("\n-----------\n")
                print("Normalized Accuracy: %0.2f" % accuracy_score(y_true, y_pred, normalize=True))
                print("Non-Normalized Accuracy: %0.2f" % accuracy_score(y_true, y_pred, normalize=False))
                f.write("Normalized Accuracy: %0.2f" % accuracy_score(y_true, y_pred, normalize=True))
                f.write("\n")
                f.write("Non-Normalized Accuracy: %0.2f" % accuracy_score(y_true, y_pred, normalize=False))
                f.close()

            f.close()
            print()
        print()
