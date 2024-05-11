#!/usr/bin/env python
# -*- coding:utf-8 -*-

import random
import time
import os
import torch

import numpy as np
import seaborn as sns
import xgboost as xgb
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def seed_everything(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


if __name__ == "__main__":
    seed_everything(seed=666)
    py_start_time = time.time()
    print('*' * 36)
    print(f"py file run at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(py_start_time))}")
    print('*' * 36)
    # =========================================
    # 超参数设置开始
    # =========================================
    # os.chdir('./insurance')

    pd.set_option("display.max_rows", 100)  # 最大行数
    pd.set_option("display.max_columns", 500)  # 最大显示列数
    pd.set_option('display.width', 200)  # 150，设置打印宽度
    pd.set_option("display.precision", 4)  # 浮点型精度

    # =========================================
    # 超参数设置结束
    # =========================================

    # 加载数据
    df = pd.read_csv('./data/insurance_claims.csv')

    # 将分类变量 'fraud_reported' 转换为数值型
    df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})

    # 删除不需要的列
    df.drop(columns='_c39', inplace=True)

    # 将所有对象类型的列转换为类别类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # noinspection PyUnreachableCode
    if 1 == 2:
        # 创建交叉表，并添加边距总计
        # insured_zip
        crosstab_with_margins = pd.crosstab(df['fraud_reported'], df['insured_hobbies'], margins=True)
        # 行归一化的交叉表
        crosstab_normalized_rows = pd.crosstab(df['fraud_reported'], df['insured_hobbies'], normalize='rows')
        print(crosstab_with_margins)
        print(crosstab_normalized_rows)

    # 定义特征 X 和目标变量 y
    # X = df.drop(columns=['fraud_reported', 'policy_bind_date', ])
    X = df.drop(columns=['fraud_reported', ])
    y = df['fraud_reported']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    # # 转换分类特征为 XGBoost 需要的格式 DMatrix
    # dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.to_list(), enable_categorical=True, )
    #
    # params_cart = {
    #     # 'num_boost_round': 1,
    #     # 'n_estimators': 30,
    #     # 'num_parallel_tree': 1,
    #     'num_parallel_tree': 1,
    #     'learning_rate': 0.08,
    #     'booster': 'gbtree',
    #     # 'booster': 'dart',
    #     # 'max_depth': 12,
    #     # 'min_child_weight': 1,
    #     'gamma': 0.1,
    #     'subsample': .9,
    #     'colsample_bytree': .9,
    #     'colsample_bynode': .9,
    #     'objective': 'binary:logistic',
    #     'tree_method': 'hist',
    #     # 'tree_method': 'exact',
    #     # 'device': "cpu",
    #     'device': "cuda",
    #     # 'enable_categorical': True,
    #     "max_cat_to_onehot": 60,
    #     'random_state': 666,
    # }
    # params_rf = params_cart.copy()
    # params_rf['num_parallel_tree'] = 30
    # params_rf['subsample'] = .8
    # params_rf['colsample_bytree'] = .8
    # params_rf['colsample_bynode'] = .8
    #
    # params_xgb = params_rf.copy()
    # params_xgb['num_parallel_tree'] = 1
    # # 定义参数
    # # print(f"the max_depth is 【{params_cart['max_depth']}】")
    #
    # # ===================================
    # # 用CART预测
    # # ===================================
    X_train_encoded = pd.get_dummies(X_train, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, drop_first=True)
    # 重新定义特征和目标变量
    y = df['fraud_reported']
    # 创建决策树分类器实例
    from sklearn.tree import DecisionTreeClassifier

    dt_classifier = DecisionTreeClassifier(random_state=666)
    dt_classifier.fit(X_train_encoded, y_train)
    # 预测
    y_pred = dt_classifier.predict(X_test_encoded)
    pred_cart = y_pred
    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("【CART】分类结果矩阵:")
    print(conf_matrix)
    print("=" * 60)
    res_clf = classification_report(y_test, y_pred, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)

    # # ===================================
    # # 用RF预测
    # # ===================================
    # 使用 XGBoost 训练模型
    xgb_rf = xgb.XGBRFClassifier(enable_categorical=True, random_state=666, ).fit(X_train, y_train)
    # 使用训练好的模型进行预测
    y_pred = np.round(xgb_rf.predict(X_test, ))
    pred_rf = y_pred
    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("【随机森林】分类结果矩阵:")
    print(conf_matrix)
    print("=" * 60)
    res_clf = classification_report(y_test, y_pred, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)
    # list(zip(xgb_model.feature_names_in_, xgb_model.feature_importances_))
    # # ===================================
    # # 用XGB预测
    # # ===================================
    # 使用 XGBoost 训练模型
    params = {
        'n_estimators': 100,
        # 'learning_rate': 0.1,
        # 'booster': 'gbtree',
        # 'booster': 'dart',
        # 'max_depth': 12,
        # 'min_child_weight': 1,
        # 'gamma': 0.1,
        # 'subsample': .9,
        # 'colsample_bytree': .9,
        # 'colsample_bynode': .9,
        # 'objective': 'binary:logistic',
        # 'tree_method': 'hist',
        # 'tree_method': 'exact',
        # 'device': "cpu",
        # 'device': "cuda",
        # 'enable_categorical': True,
        # "max_cat_to_onehot": 60,
        # 'random_state': 666,
    }
    # params = {
    #     'n_estimators': 30,
    #     'learning_rate': 0.05,
    #     'max_depth': 9,
    #     'min_child_weight': 1,
    #     'gamma': 0.1,
    #     'subsample': 0.8,
    #     'colsample_bytree': 0.8,
    #     'objective': 'binary:logistic',
    #     'tree_method': 'hist',
    #     # 'tree_method': "hist",
    #     'booster': 'gbtree',
    #     'device': "cuda",
    #     # 'use_softplus': False,
    #     # 'enable_categorical': True,
    #     # 'random_state': 666,
    #     # 'eval_metric': 'mlogloss',
    #     # 'early_stopping_rounds': 10,
    # }
    # xgb建模
    xgb_cls = xgb.XGBClassifier(**params, enable_categorical=True, random_state=666, ).fit(X_train, y_train)
    # 预测测试集
    y_pred = np.round(xgb_cls.predict(X_test))
    pred_xgb = y_pred
    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("【XGBoost】分类结果矩阵:")
    print(conf_matrix)
    print("=" * 60)
    res_clf = classification_report(y_test, y_pred, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)
    # ===================================
    # 绘制ROC曲线
    # ===================================
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt

    # 假设pred_cart, pred_rf, pred_xgb是三个分类器预测的正类概率，y_label是真实标签
    # 注意：这里我们使用predict_proba方法来获取正类的概率，对于二进制分类问题，通常是最后一列

    # 计算每个分类器的ROC曲线和AUC值
    fpr_cart, tpr_cart, _ = roc_curve(y_test, pred_cart)
    roc_auc_cart = auc(fpr_cart, tpr_cart)

    fpr_rf, tpr_rf, _ = roc_curve(y_test, pred_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)

    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, pred_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

    # 绘制ROC曲线
    plt.figure()
    lw = 2
    plt.plot(fpr_cart, tpr_cart, color='darkorange', lw=lw, label='CART ROC curve (area = %0.2f)' % roc_auc_cart)
    plt.plot(fpr_rf, tpr_rf, color='blue', lw=lw, label='RF ROC curve (area = %0.2f)' % roc_auc_rf)
    plt.plot(fpr_xgb, tpr_xgb, color='green', lw=lw, label='XGB ROC curve (area = %0.2f)' % roc_auc_xgb)

    # 绘制对角线，表示随机猜测的基准
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

    # 设置图表的标题和坐标轴标签
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver Operating Characteristic to multi-class')
    plt.legend(loc="lower right")

    # 显示图表
    plt.show()

    # # ===================================
    # # 绘制特征重要性
    # # ===================================
    # xgb.plot_importance(xgb_model, importance_type='weight', title='Feature Importance', grid=False, max_num_features=20, )
    xgb.plot_importance(xgb_cls, title='Feature Importance weight')
    plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance weight', importance_type='weight')
    # plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance total_gain', importance_type='111')
    # plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance total_cover', importance_type='total_cover')
    # plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance gain', importance_type='gain')
    # plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance cover', importance_type='cover')
    # plt.show()
    # xgb.plot_importance(xgb_cls, title='Feature Importance', importance_type='frequency')
    # plt.show()

    # importance_types = ['weight', 'total_gain', 'total_cover', 'gain', 'cover']
    # f = 'gain'
    # # 按重要性降序排列
    # # importance_df = 1
    # tmp_df = pd.DataFrame(list(xgb_cls.get_booster().get_score(importance_type=f).items()),
    #                       columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    # tmp_df['Importance'] = tmp_df['Importance'] /tmp_df['Importance'].sum()
    # tmp_df
    # 从目前的结果来看xgb_cls.feature_importances_使用的是归一化的gain类型的重要性。

    # 特征重要性
    # 获取特征重要性
    feature_importances = xgb_cls.feature_importances_
    # 创建一个DataFrame来存储特征名称和重要性
    feature_names = xgb_cls.feature_names_in_
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # 按重要性降序排列
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm', hue='Feature', legend=False)
    # 设置标题和轴标签
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    # 显示图形
    plt.show()

    py_end_time = time.time()
    print('*' * 36)
    print(f"py file run at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(py_end_time))}")
    print(f'Running time: {(py_end_time - py_start_time) / 60:.2f} minutes')
    print('*' * 36)
