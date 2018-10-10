import sklearn.naive_bayes as nb
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('spambase.data')
spam_features = data.drop('spam_or_not',axis=1)
spam_labels = data.loc[:,'spam_or_not']

bayes = {'Gaussian': nb.GaussianNB(),'Multinomial': nb.MultinomialNB(),'Complement': nb.ComplementNB(),'Bernoulli': nb.BernoulliNB()}
for k in bayes.keys():
    predictions = bayes[k].fit(spam_features, spam_labels).predict(spam_features)
    misses = (spam_labels!=predictions).sum()
    false_positive = misses-(predictions[predictions!=spam_labels]==1).sum()
    false_negative = misses-false_positive
    print('{} Method Mislabelled {} points {}% {} false positive {} false negative'.format(k,misses,misses/spam_labels.shape[0]*100,false_positive,false_negative))



