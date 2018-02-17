from sklearn import neighbors, neural_network, svm

# define the data set
X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43]
    ]

y = [
    'male',
    'male',
    'female',
    'female',
    'male',
    'male',
    'female',
    'female',
    'female',
    'male',
    'male'
    ]

# write names to a list so we can output the best one
names = [
    'KNeighborsClassifier',
    'MLP Classifier',
    'SupportVectorClassifier'
    ]

# set up classifiers
classifiers = [
    neighbors.KNeighborsClassifier(),
    neural_network.MLPClassifier(),
    svm.SVC(probability=True)
    ]

best_clf = { 'name': None, 'value': None }

# fit data and compare classifiers by calculating the difference
# between the probabilities for the predicted classes
for clf in classifiers:
    clf.fit(X, y)
    probability = clf.predict_proba([[190, 70, 43]])[0]
    #print(probability, max(probability))
    diff = abs(probability[0] - probability[1])
    if diff > best_clf['value']:
        best_clf['name'] = names[classifiers.index(clf)]
        best_clf['value'] = max(probability)

print('The best classifier is the {} with a confidence of {:.5f}'.format(best_clf['name'], best_clf['value']))
