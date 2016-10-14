#!/usr/bin/python
# -*-coding: utf-8-*-

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.stats import pearsonr
from minepy import MINE

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import f_regression

from sklearn.decomposition import PCA
from sklearn.lda import LDA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RandomizedLasso
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
import LR


def mic(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    m = MINE()
    m.compute_score(x, y)
    return (m.mic(), 0.5)


class FeaturePreproces(object):

    def __init__(self):
        """
        purpose: Constructor
        """
        #feature
        self.columns = ["price", "quantity", "orders", "price_7", "quantity_7", "orders_7", "price_30", "quantity_30",
        "orders_30", "price_total", "quantity_total", "orders_total", "score", "grade_count", "ipv_7",
          "ipv_30", "ipv_uv_7", "ipv_uv_30", "shop_favor_7", "shop_favor_30",
          "item_favor_7", "item_favor_30", "pay_order_cnt_7", "pay_order_cnt_30", "refund_order_cnt_7",
          "refund_order_cnt_30", "refund_rate_7", "refund_rate_30", "ipv_1", "ipv_uv_1",
          "shop_favor_1", "item_favor_1", "pay_order_cnt_1", "refund_order_cnt_1", "refund_rate_1",
          "im_session_7", "im_reply_session_7", "category_cnt", "ship_ratio_24h", "ship_ratio_72h",
          "order_days_30", "ship_days_30", "item_add_30", "item_add_7", "tsp_ratio_7",
          "tsp_ratio_30", "act_days_30", "reply_10min", "reply_30min", "reply_60min",
          "item_cnt", "sku_cnt", "is_cpc", "is_baoyou", "is_wlyth",
          "is_cps", "is_tb_move", "is_guaranteed", "is_ppgf",
          "is_ppsq", "is_zjrz"]

    def load_data(self, file_name):
        """
        purpose: Load data
        :return (X, y)
        """
        #df = pd.read_csv("train.csv")  #read csv file
        df = pd.read_excel(file_name)    #read xls file
        # print df.describe()

        print df[self.columns].describe()

        X = df[self.columns].values
        y = df['target'].values

        return (X, y)

    def preprocessing(self, X, method):
        """
        purpose: preprocess
        input:  X: data
                method: method
        """
        # Standard Scale
        if method == 'StandardScaler':
            X_new = StandardScaler().fit_transform(X)

        # scale, return[0-1]
        elif method == 'MinMaxScaler':
            X_new = MinMaxScaler().fit_transform(X)

        # Normalize
        elif method == 'Normalizer':
            X_new = Normalizer().fit_transform(X)

        # binarize
        elif method == 'Binarizer':
            X_new = Binarizer(threshold=3).fit_transform(X)

        # One Hot Encode
        elif method == 'OneHotEncoder':
            new_data = OneHotEncoder().fit_transform()

        # missing data imputation
        # missing_value is the style of missing data,default is NaN
        # strategy is fill mode,default is mean
        elif method == 'Imputer':
            X_new = Imputer().fit_transform(np.vstack((np.array([np.nan, np.nan, np.nan, np.nan]), X)))

        # polynomial transform,default degree is 2
        elif method == 'PolynomialFeatures':
            X_new = PolynomialFeatures().fit_transform(X)

        # Custom transformers
        # first param is the function
        elif method == 'FunctionTransformer':
            X_new = FunctionTransformer(np.log1p).fit_transform(X)

        return X_new

    def feature_selection(self, X, y, method):
        """
        purpose:    select feature
        input:  X:train data
                y:lable
                method: uesed method
        return:
        """
        X_indices = np.arange(X.shape[-1])

        score = []

        # Removing features with low variance

        # correlation coefficient
        # SelectKBest(lambda X,Y: np.array(map(lambda x: pearsonr(x, Y), X.T)).T, k=2).fit_transform(data, target)

        # mutual information
        # SelectKBest(lambda X, Y: array(map(lambda x: mic(x, Y), X.T)).T, k=2).fit_transform(data, target)

        # Univariate feature selection (for classification)
        if method == 'chi-squared':
            skb = SelectKBest(chi2)
            skb.fit_transform(X, y)
            score = skb.scores_

        # Univariate feature selection (for regression)
        if method == 'f_regression':
            skb = SelectKBest(f_regression)
            skb.fit_transform(X, y)
            score = skb.scores_

        # L1-based feature selection (for classification)
        if method == 'LinearSVC':
            lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
            sfm = SelectFromModel(lsvc, prefit=True)
            X_new = sfm.transform(X)

        # L1-based feature selection (for regression)
        elif method == 'LassoCV':
            lasso = LassoCV().fit(X, y)
            score = lasso.coef_
            sfm = SelectFromModel(lasso, threshold=0.25, prefit=True)
            X_new = sfm.transform(X)

        # Tree-based feature selection (for classification)
        elif method == 'ExtraTreesClassifier':
            clf = ExtraTreesClassifier()
            clf = clf.fit(X, y)
            print clf.feature_importances_
            sfm = SelectFromModel(clf, threshold=0.25, prefit=True)
            X_new = sfm.transform(X)

        # Tree-based feature selection (for regression)
        elif method == 'ExtraTreesRegressor':
            clf = ExtraTreesRegressor()
            clf = clf.fit(X, y)
            score = clf.feature_importances_
            sfm = SelectFromModel(clf, threshold=0.25, prefit=True)
            X_new = sfm.transform(X)

        # Tree-based feature selection (for classifier)
        elif method == 'GradientBoostingClassifier':
            clf = GradientBoostingClassifier(learning_rate=0.01)
            clf = clf.fit(X, y)
            score = clf.feature_importances_
            sfm = SelectFromModel(clf, threshold=0.25, prefit=True)
            X_new = sfm.transform(X)

        # Tree-based feature selection (for regression)
        elif method == 'GradientBoostingRegressor':
            clf = GradientBoostingRegressor(learning_rate=0.01)
            clf = clf.fit(X, y)
            score = clf.feature_importances_
            sfm = SelectFromModel(clf, threshold=0.25, prefit=True)
            X_new = sfm.transform(X)

        # Print the feature ranking
        indices = np.argsort(score)[::-1]
        print("Feature ranking:")
        for f in X_indices:
            print("feature %d: %s  (%f)" % (indices[f], self.columns[indices[f]], score[indices[f]]))

        #draw plot
        plt.figure()
        # plt.bar(indices, score, width=0.2, color='r')
        plt.barh(indices, score, height=0.2, color='r')
        plt.title(method)
        plt.xlabel("score")
        plt.ylabel("feature")
        plt.grid(axis='x')
        plt.show()

        pass

    def dimension_reduction(self, X, y):
        """
        purpose: reduce dimension
        :return:
        """
        # PCA, n_components is numbers of principal component
        X_new = PCA(n_components=2).fit_transform(X)

        # LDA, n_components is the dimension after reduce
        X_new = LDA(n_components=2).fit_transform(X, y)

        pass

if __name__ == '__main__':
    file_name = "train_data.xls"

    feature_preproces = FeaturePreproces()
    (X, y) = feature_preproces.load_data(file_name)
    # feature_preproces.preprocessing(X)
    feature_preproces.feature_selection(X, y, 'chi-squared')
    #feature_preproces.dimension_reduction(data, target)
