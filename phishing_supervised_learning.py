# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 15:42:58 2017

@author: JTay

Original repo found here: https://github.com/JonathanTay/CS-7641-assignment-1

Modified for phishing dataset by Bryce Turner
"""

import numpy as np
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier as knnC
import pandas as pd
from collections import defaultdict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifie
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.utils import compute_sample_weight
from sklearn.ensemble import AdaBoostClassifier
import os
from time import clock


class dtclf_pruned(dtclf):
    def remove_subtree(self, root):
        '''Clean up'''
        tree = self.tree_
        visited, stack = set(), [root]
        while stack:
            v = stack.pop()
            visited.add(v)
            left = tree.children_left[v]
            right = tree.children_right[v]
            if left >= 0:
                stack.append(left)
            if right >= 0:
                stack.append(right)
        for node in visited:
            tree.children_left[node] = -1
            tree.children_right[node] = -1
        return

    def prune(self):
        C = 1 - self.alpha
        if self.alpha <= -1:  # Early exit
            return self
        tree = self.tree_
        bestScore = self.score(self.valX, self.valY)
        candidates = np.flatnonzero(tree.children_left >= 0)
        for candidate in reversed(candidates):  # Go backwards/leaves up
            if tree.children_left[candidate] == tree.children_right[candidate]:  # leaf node. Ignore
                continue
            left = tree.children_left[candidate]
            right = tree.children_right[candidate]
            tree.children_left[candidate] = tree.children_right[candidate] = -1
            score = self.score(self.valX, self.valY)
            if score >= C * bestScore:
                bestScore = score
                self.remove_subtree(candidate)
            else:
                tree.children_left[candidate] = left
                tree.children_right[candidate] = right
        assert (self.tree_.children_left >= 0).sum() == (self.tree_.children_right >= 0).sum()
        return self

    def fit(self, X, Y, sample_weight=None, check_input=True, X_idx_sorted=None):
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        self.trgX = X.copy()
        self.trgY = Y.copy()
        self.trgWts = sample_weight.copy()
        sss = ms.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
        for train_index, test_index in sss.split(self.trgX, self.trgY):
            self.valX = self.trgX[test_index]
            self.valY = self.trgY[test_index]
            self.trgX = self.trgX[train_index]
            self.trgY = self.trgY[train_index]
            self.valWts = sample_weight[test_index]
            self.trgWts = sample_weight[train_index]
        super().fit(self.trgX, self.trgY, self.trgWts, check_input, X_idx_sorted)
        self.prune()
        return self

    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_split=1e-7,
                 class_weight=None,
                 presort=False,
                 alpha=0):
        super(dtclf_pruned, self).__init__(
            criterion=criterion,
            splitter=splitter,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            class_weight=class_weight,
            random_state=random_state,
            min_impurity_split=min_impurity_split,
            presort=presort)
        self.alpha = alpha

    def numNodes(self):
        assert (self.tree_.children_left >= 0).sum() == (self.tree_.children_right >= 0).sum()
        return (self.tree_.children_left >= 0).sum()

    #Functions for computing accuracy and outputting basic results
def balanced_accuracy(truth, pred):
    wts = compute_sample_weight('balanced', truth)
    return accuracy_score(truth, pred, sample_weight=wts)


scorer = make_scorer(balanced_accuracy)


def basicResults(clfObj, trgX, trgY, tstX, tstY, params, clf_type=None, dataset=None):
    np.random.seed(55)
    if clf_type is None or dataset is None:
        raise Exception("No data set!")
    cv = ms \
        .GridSearchCV(clfObj, n_jobs=1, param_grid=params, refit=True, verbose=10, cv=5, scoring=scorer)
    cv.fit(trgX, trgY)
    cv.fit(trgX, trgY)
    regTable = pd.DataFrame(cv.cv_results_)
    outdir = './output_iter'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    fullname = os.path.join(outdir, '{}_{}_reg.csv'.format(clf_type, dataset))
    regTable.to_csv(fullname, index=False)
    test_score = cv.score(tstX, tstY)
    fullname = os.path.join(outdir, '{}_{}_test_results.csv'.format(clf_type, dataset))
    with open(fullname, 'a') as f:
        f.write('{},{},{},{}\n'.format(clf_type, dataset, test_score, cv.best_params_))
    N = trgY.shape[0]
    curve = ms.learning_curve(cv.best_estimator_, trgX, trgY, cv=5,
                              train_sizes=[50, 100] + [int(N * x / 10) for x in range(1, 8)], verbose=10,
                              scoring=scorer)
    curve_train_scores = pd.DataFrame(index=curve[0], data=curve[1])
    curve_test_scores = pd.DataFrame(index=curve[0], data=curve[2])
    fullname = os.path.join(outdir, '{}_{}_LC_train.csv'.format(clf_type, dataset))
    curve_train_scores.to_csv(fullname)
    fullname = os.path.join(outdir, '{}_{}_LC_test.csv'.format(clf_type, dataset))
    curve_test_scores.to_csv(fullname)
    return cv


def makeTimingCurve(X, Y, clf, clfName, dataset):
    out = defaultdict(dict)
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        X_train, X_test, y_train, y_test = ms.train_test_split(X, Y, test_size=frac, random_state=42)
        st = clock()
        np.random.seed(55)
        clf.fit(X_train, y_train)
        out['train'][frac] = clock() - st
        st = clock()
        clf.predict(X_test)
        out['test'][frac] = clock() - st
        print(clfName, dataset, frac)
    out = pd.DataFrame(out)
    out.to_csv('./output_iter/{}_{}_timing.csv'.format(clfName, dataset))
    return


#KNN
phish = pd.read_csv('phish.csv')
phishX = phish.drop('Result', 1).copy().values
phishY = phish['Result'].copy().values

phish_trgX, phish_tstX, phish_trgY, phish_tstY = ms.train_test_split(phishX, phishY, test_size=0.3, random_state=0, stratify=phishY)


d = phishX.shape[1]
hiddens_phish = [(h,)*l for l in [1,2,3] for h in [d,d//2,d*2]]
alphas = [10**-x for x in np.arange(1,9.01,1/2)]


pipeA = Pipeline([('Scale',StandardScaler()),                
                 ('KNN',knnC())])  


params_phish= {'KNN__metric':['manhattan','euclidean','chebyshev'],'KNN__n_neighbors':np.arange(1,51,3),'KNN__weights':['uniform','distance']}

phish_clf = basicResults(pipeA, phish_trgX, phish_trgY, phish_tstX, phish_tstY, params_phish, 'KNN', 'phish_sites')

phish_final_params=phish_clf.best_params_

pipeA.set_params(**phish_final_params)
makeTimingCurve(phishX, phishY, pipeA, 'KNN', 'phish')

#Decision Trees
def DTpruningVSnodes(clf, alphas, trgX, trgY, dataset):
    '''Dump table of pruning alpha vs. # of internal nodes'''
    out = {}
    for a in alphas:
        clf.set_params(**{'DT__alpha': a})
        clf.fit(trgX, trgY)
        out[a] = clf.steps[-1][-1].numNodes()
        print(dataset, a)
    out = pd.Series(out)
    out.index.name = 'alpha'
    out.name = 'Number of Internal Nodes'
    out.to_csv('./output_final/DT_{}_nodecounts.csv'.format(dataset))

    return


# Load Data

phish_trgX, phish_tstX, phish_trgY, phish_tstY = ms.train_test_split(phishX, phishY, test_size=0.3, random_state=0,
                                                                     stratify=phishY)

# Search for good alphas
alphas = [-1, -1e-3, -(1e-3) * 10 ** -0.5, -1e-2, -(1e-2) * 10 ** -0.5, -1e-1, -(1e-1) * 10 ** -0.5, 0,
          (1e-1) * 10 ** -0.5, 1e-1, (1e-2) * 10 ** -0.5, 1e-2, (1e-3) * 10 ** -0.5, 1e-3]
# alphas=[0]
pipeM = Pipeline([('Scale', StandardScaler()),
                  ('DT', dtclf_pruned(random_state=55))])

params = {'DT__criterion': ['gini', 'entropy'], 'DT__alpha': alphas, 'DT__class_weight': ['balanced']}

phish_clf = basicResults(pipeM, phish_trgX, phish_trgY, phish_tstX, phish_tstY, params, 'DT', 'phish')

phish_final_params = phish_clf.best_params_

pipeM.set_params(**phish_final_params)
makeTimingCurve(phishX, phishY, pipeM, 'DT', 'phishsites')

#SVM

phishX = phish.drop('Result', 1).copy().values
phishY = phish['Result'].copy().values


phish_trgX, phish_tstX, phish_trgY, phish_tstY = ms.train_test_split(phishX, phishY, test_size=0.3, random_state=0,
                                                                     stratify=phishY)
N_phish = phish_trgX.shape[0]

alphas = [10 ** -x for x in np.arange(1, 9.01, 1 / 2)]

# RBF SVM
# gamma_fracsA = np.arange(0.2, 2.1, 0.2)
# gamma_fracsM = np.arange(0.05, 1.01, 0.1)

#
pipeSVM = Pipeline([('Scale',StandardScaler()),
                ('SVM',SGDClassifier(loss='hinge',l1_ratio=0,penalty='l2',class_weight='balanced',random_state=55))])

params_phish = {'SVM__alpha': alphas, 'SVM__n_iter': [int((1e6 / N_phish) / .8) + 1], 'SVM__gamma_frac': gamma_fracsA}

#

phish_clf = basicResults(pipeSVM, phish_trgX, phish_trgY, phish_tstX, phish_tstY, params_phish, 'SVM_LIN', 'phish')


phish_final_params = phish_clf.best_params_
phish_OF_params = phish_final_params.copy()
phish_OF_params['SVM__alpha'] = 1e-16


pipeSVM.set_params(**phish_final_params)
makeTimingCurve(phishX, phishY, pipeSVM, 'SVM_LIN', 'phish')

#Boosting

alphas = [-1, -1e-3, -(1e-3) * 10 ** -0.5, -1e-2, -(1e-2) * 10 ** -0.5, -1e-1, -(1e-1) * 10 ** -0.5, 0,
          (1e-1) * 10 ** -0.5, 1e-1, (1e-2) * 10 ** -0.5, 1e-2, (1e-3) * 10 ** -0.5, 1e-3]

phish_base = dtclf_pruned(criterion='entropy', class_weight='balanced', random_state=55)

# paramsA= {'Boost__n_estimators':[1,2,5,10,20,30,40,50],'Boost__learning_rate':[(2**x)/100 for x in range(8)]+[1]}
paramsBoost = {'Boost__n_estimators': [1, 2, 5, 10, 20, 30, 45, 60, 80, 100],
              'Boost__base_estimator__alpha': alphas}

phish_booster = AdaBoostClassifier(algorithm='SAMME', learning_rate=1, base_estimator=phish_base, random_state=55)

pipeBoost = Pipeline([('Scale', StandardScaler()),
                      ('Boost', phish_booster)])

#

phish_clf = basicResults(pipeBoost, phish_trgX, phish_trgY, phish_tstX, phish_tstY, paramsBoost, 'Boost', 'phish')



