import time

import os
import pandas as pd
import xgboost as xgb
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
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
    df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})
    df.drop(columns='_c39', inplace=True)
    for col in df.columns:
        if df[col].dtypes == 'object':
            df[col] = df[col].astype('category')

    X, y = df.drop(columns='fraud_reported'), df['fraud_reported']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    params = {
        'n_estimators': 30,
        'learning_rate': 0.05,
        'max_depth': 9,
        'min_child_weight': 1,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'tree_method': 'gpu_hist',
        # 'tree_method': "hist",
        'booster': 'gbtree',
        'device': "cuda",
        'use_softplus': False,
        'enable_categorical': True,
        'random_state': 666,
        # 'eval_metric': 'mlogloss',
        # 'early_stopping_rounds': 10,
    }
    # xgb建模
    xgb_cls = xgb.XGBClassifier(**params, )
    # 训练模型
    xgb_cls.fit(X_train, y_train)
    # 预测测试集
    y_pred = xgb_cls.predict(X_test)
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')  # 平均方式可以是 'micro', 'macro', 'weighted', 'samples'
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_test, y_pred)
    # 打印评估指标
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

    print("=" * 60)
    print("分类结果矩阵:\n")

    print(classification_report(y_test, y_pred))
    print("=" * 60)

    # 特征重要性
    # 获取特征重要性
    feature_importances = xgb_cls.feature_importances_
    # 创建一个DataFrame来存储特征名称和重要性
    feature_names = X_train.columns
    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
    # 按重要性降序排列
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    # 绘制条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df, palette='coolwarm')
    # 设置标题和轴标签
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Features')
    # 显示图形
    plt.show()

    # =========================================
    # 大面积的损坏和保险欺诈显著相关，不然就没必要欺诈了。很真实。
    # =========================================
    # 创建交叉表
    crosstabulation = pd.crosstab(df['fraud_reported'], df['incident_severity'], margins=True)
    # 打印交叉表
    print(crosstabulation)

    py_end_time = time.time()
    print('*' * 36)
    print(f"py file run at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(py_end_time))}")
    print(f'Running time: {(py_end_time - py_start_time) / 60:.2f} minutes')
    print('*' * 36)
