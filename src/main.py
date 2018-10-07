import pandas as pd
from sklearn import metrics, preprocessing
import time


def naive_bayes_classifier(train_x, train_y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB(alpha=1.0)
    model.fit(train_x, train_y)
    return model


# KNN Classifier
def knn_classifier(train_x, train_y):
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(train_x, train_y)
    return model


# Logistic Regression Classifier
def logistic_regression_classifier(train_x, train_y):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(penalty='l2')
    model.fit(train_x, train_y)
    return model


# Random Forest Classifier
def random_forest_classifier(train_x, train_y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)
    model.fit(train_x, train_y)
    return model


# Decision Tree Classifier
def decision_tree_classifier(train_x, train_y):
    from sklearn import tree
    model = tree.DecisionTreeClassifier(criterion="entropy")
    model.fit(train_x, train_y)
    return model


# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_x, train_y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=200)
    model.fit(train_x, train_y)
    return model


# SVM Classifier
def svm_classifier(train_x, train_y):
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    model.fit(train_x, train_y)
    return model


# SVM Classifier using cross validation
def svm_cross_validation(train_x, train_y):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    model = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(model, param_grid, n_jobs=1, verbose=1)
    grid_search.fit(train_x, train_y)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print(para, val)
    model = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    model.fit(train_x, train_y)
    return model


if __name__ == '__main__':
    test_classifiers = [
        'Naive Bayes',
        'Logistic Regression',
        'KNN',
        'Decision Tree',
        'Random Forest',
        'SVM'
    ]
    classifiers = {
        'Naive Bayes': naive_bayes_classifier,
        'KNN': knn_classifier,
        'Logistic Regression': logistic_regression_classifier,
        'Random Forest': random_forest_classifier,
        'Decision Tree': decision_tree_classifier,
        'SVM': svm_classifier,
        'SVM Cross Validation': svm_cross_validation
    }

    train_data_list = [
        pd.read_csv("../data/small-csv/train-best10.csv", header=None),
        pd.read_csv("../data/medium-csv/train-best50.csv", header=None),
        pd.read_csv("../data/large-csv/train-best200.csv", header=None)
    ]
    dev_data_list = [
        pd.read_csv("../data/small-csv/dev-best10.csv", header=None),
        pd.read_csv("../data/medium-csv/dev-best50.csv", header=None),
        pd.read_csv("../data/large-csv/dev-best200.csv", header=None)
    ]

    test_data_list = [
        pd.read_csv("../data/small-csv/test-best10.csv", header=None),
        pd.read_csv("../data/medium-csv/test-best50.csv", header=None),
        pd.read_csv("../data/large-csv/test-best200.csv", header=None)
    ]

    models = [[] for i in range(0, len(train_data_list))]
    for i in range(0, len(train_data_list)):

        train_data = train_data_list[i]
        train_x = train_data.values[:, 2:-1]
        # train_x = preprocessing.minmax_scale(train_x, feature_range=(0, 1))
        train_y = train_data.values[:, -1]

        dev_data = dev_data_list[i]
        dev_x = dev_data.values[:, 2:-1]
        # dev_x = preprocessing.minmax_scale(dev_x, feature_range=(0, 1))
        dev_y = dev_data.values[:, -1]

        test_data = test_data_list[i]
        test_x = test_data.values[:, 2:-1]
        # test_x = preprocessing.minmax_scale(test_x, feature_range=(0, 1))

        for classifier in test_classifiers:
            print("===== " + classifier + " " + str(i) + " =====")
            start_time = time.time()
            model = classifiers[classifier](train_x, train_y)
            models[i].append(model)
            predict_y = model.predict(dev_x)
            print("confusion matrix: ")
            print(metrics.confusion_matrix(dev_y, predict_y))
            print('accuracy: %.4f' % (metrics.accuracy_score(dev_y, predict_y)))
            print("report: ")
            print(metrics.classification_report(dev_y, predict_y, digits=4))
            print('total time: %.3fs' % (time.time() - start_time))
            print()

    # Choose Random Forest to predict
    test_y_s = models[0][4].predict(test_data_list[0].values[:, 2:-1])
    test_y_m = models[1][4].predict(test_data_list[1].values[:, 2:-1])
    test_y_l = models[2][4].predict(test_data_list[2].values[:, 2:-1])

    with open('../output/labels-test-best10.csv', 'w') as f0:
        for line in test_y_s:
            f0.write(line + "\n")

    with open('../output/labels-test-best50.csv', 'w') as f1:
        for line in test_y_m:
            f1.write(line + "\n")

    with open('../output/labels-test-best200.csv', 'w') as f2:
        for line in test_y_l:
            f2.write(line + "\n")

    # with open('../output/test-best10.csv', 'w') as f0:
    #     with open("../data/small-csv/test-best10.csv", "r") as g0:
    #         for i in range(len(test_y_s)):
    #             f0.write(g0.readline().replace("\n", "").replace("?", test_y_s[i]))
    #             f0.write("\n")
    #
    # with open('../output/test-best50.csv', 'w') as f1:
    #     with open("../data/medium-csv/test-best50.csv", "r") as g1:
    #         for i in range(len(test_y_m)):
    #             f1.write(g1.readline().replace("\n", "").replace("?", test_y_m[i]))
    #             f1.write("\n")
    #
    # with open('../output/test-best200.csv', 'w') as f2:
    #     with open("../data/large-csv/test-best200.csv", "r") as g2:
    #         for i in range(len(test_y_l)):
    #             f2.write(g2.readline().replace("\n", "").replace("?", test_y_l[i]))
    #             f2.write("\n")

    print("Done.\n")
