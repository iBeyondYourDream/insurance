
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


def model_plot_trees(clf, features, num_trees=0):
    """绘制树模型"""
    # 保存特征名称到fmap文件，用于图形绘制
    with open('./cache/xgb.fmap', 'w', encoding="utf-8") as fmap:
        for k, ft in enumerate(features):
            fmap.write(''.join(str([k, ft, 'p']) + '\n'))

    # 构建条件节点和叶子节点的形状、填充颜色
    c_node_params = {'shape': 'box',
                     'style': 'filled,rounded',
                     'fillcolor': '#78bceb'
                     }
    l_node_params = {'shape': 'box',
                     'style': 'filled',
                     'fillcolor': '#e48038'
                     }

    # 树模型绘制：有向图
    # 绘制和保存第num_trees+1棵树，num_trees为树的序号
    digraph = xgb.to_graphviz(clf, num_trees=num_trees, condition_node_params=c_node_params,
                              leaf_node_params=l_node_params, fmap='./cache/xgb.fmap')
    # digraph.format = 'png'
    digraph.view('./results/xgb_trees')

    # 分别绘制子图，不保存
    # for i in range(n.get('n')):
    #     xgb.plot_tree(clf, num_trees=i, condition_node_params=c_node_params,
    #                   leaf_node_params=l_node_params, fmap='xgb.fmap')
    #     plt.show()


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

    # 将分类变量 'fraud_reported' 转换为数值型
    df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})

    # 删除不需要的列
    df.drop(columns='_c39', inplace=True)

    # 将所有对象类型的列转换为类别类型
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category')

    # 定义特征 X 和目标变量 y
    X = df.drop(columns=['fraud_reported', 'policy_bind_date', ])
    y = df['fraud_reported']

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    # 转换分类特征为 XGBoost 需要的格式 DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=X_train.columns.to_list(), enable_categorical=True, )

    params = {
        'num_boost_round': 1,
        # 'n_estimators': 30,
        'num_parallel_tree': 1,
        'learning_rate': 0.05,
        'booster': 'gbtree',
        'max_depth': max(int(len(X_train.columns) * 0.6), 9),
        'min_child_weight': 1,
        'gamma': 0.1,
        'subsample': 1,
        'colsample_bytree': 1,
        'colsample_bynode': 1,
        'objective': 'binary:logistic',
        'tree_method': 'hist',
        'device': "cuda",
        'enable_categorical': True,
        'random_state': 666,
    }
    # 定义参数
    print(f"the max_depth is 【{params['max_depth']}】")

    # 使用 XGBoost 训练模型
    xgb_model = xgb.train(params, dtrain, )

    # 使用训练好的模型进行预测
    y_pred = xgb_model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns.to_list(), enable_categorical=True, ))
    y_pred = np.round(y_pred)
    # y_pred_prob = xgb_model.predict(xgb.DMatrix(X_test, feature_names=X_test.columns.to_list(), enable_categorical=True, ), output_margin=True)
    # dir(xgb_model)
    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)

    print("分类结果矩阵:\n")
    print(conf_matrix)
    print("=" * 60)

    print(classification_report(y_test, y_pred))
    print("=" * 60)

    model_plot_trees(xgb_model, xgb_model.feature_names, num_trees=0)

    py_end_time = time.time()
    print('*' * 36)
    print(f"py file run at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(py_end_time))}")
    print(f'Running time: {(py_end_time - py_start_time) / 60:.2f} minutes')
    print('*' * 36)
