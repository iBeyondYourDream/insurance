#!/usr/bin/env python
# coding: utf-8
import json
import os
import pickle
import re
import random
import requests
import torch

import numpy as np
import pandas as pd
import xgboost as xgb

from lime import lime_tabular
from pynvml import nvmlInit, nvmlDeviceGetMemoryInfo, nvmlDeviceGetHandleByIndex
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import lift_score
from tqdm import tqdm
from sklearn.metrics import classification_report, mean_squared_error, roc_auc_score, roc_curve
from http import HTTPStatus


def calculate_ks_score(y_true, y_scores):
    """
    计算KS得分
    :param y_true: 真实标签，通常为0和1的二元标签
    :param y_scores: 模型预测的概率或得分
    :return: KS得分
    """
    y_true, y_scores = np.array(y_true), np.array(y_scores)
    # 按照预测得分排序，并获取对应的真实标签
    sorted_indices = np.argsort(y_scores)
    y_true_sorted = y_true[sorted_indices]
    # 计算累积正负样本数
    cum_positives = np.cumsum(y_true_sorted)
    cum_negatives = np.cumsum(1 - y_true_sorted)
    # 计算累积比率
    cum_positives_ratio = cum_positives / cum_positives[-1]
    cum_negatives_ratio = cum_negatives / cum_negatives[-1]
    # 计算KS值
    ks_values = np.abs(cum_positives_ratio - cum_negatives_ratio)
    ks_score = np.max(ks_values)
    # 可选的：使用scipy的ks_2samp函数计算p值（但这通常用于两个独立样本的比较）
    # ks_stat, p_value = ks_2samp(y_true[y_scores < np.median(y_scores)], y_true[y_scores >= np.median(y_scores)])
    return ks_score


def seed_everything(seed=666):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)


if __name__ == "__main__":
    # =========================================
    # 超参数设置开始
    # =========================================
    # os.chdir('./insurance')
    # os.getcwd()
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    seed_everything(seed=666)

    pd.set_option("display.max_rows", 100)  # 最大行数
    pd.set_option("display.max_columns", 500)  # 最大显示列数
    pd.set_option('display.width', 200)  # 150，设置打印宽度
    pd.set_option("display.precision", 4)  # 浮点型精度

    # 检查CUDA是否可用，并据此设置设备
    nvmlInit()
    tmp = [nvmlDeviceGetMemoryInfo(nvmlDeviceGetHandleByIndex(_)).used for _ in range(torch.cuda.device_count())]
    gpu_id = tmp.index(min(tmp))
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"This py file run on the device: 【{device}】.")

    model_name = 'multimodalmultimodal'
    model_path = f"./results/model_{model_name}_parameters.pth"
    print(f'the model is 【{model_name}】')
    # =========================================
    # 超参数设置结束
    # =========================================

    # ==========================1. 加载输入数据==========================
    # 加载数据
    df = pd.read_csv('./data/insurance_claims.csv')
    # 将分类变量 'fraud_reported' 转换为数值型
    df['fraud_reported'] = df['fraud_reported'].map({'N': 0, 'Y': 1})
    df['incident_date'] = pd.to_datetime(df['incident_date'])
    # 创建一个月份数字到英文简写的映射字典
    months_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug',
                  9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec', }
    # 使用map()函数将月份数字映射到英文简写
    df['incident_month'] = df['incident_date'].dt.month.map(months_map)
    # 删除不需要的列
    df.drop(columns='_c39', inplace=True)

    # ==========================2. 训练的小模型==========================
    # 定义特征 X 和目标变量 y，删掉不需要的列
    # X = df.drop(columns=['fraud_reported', 'policy_bind_date', ])
    X = df.drop(columns=['fraud_reported', 'policy_bind_date', 'incident_location', 'incident_date'])
    y = df['fraud_reported']

    # X = pd.get_dummies(data=X,drop_first=False,dtype=int)
    list_LabelEncoder = []
    # 将所有对象类型的列转换为类别类型
    for col in X.columns:
        if X[col].dtype == 'object' and col != 'fraud_reported':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            X[col] = X[col].astype('category')
            list_LabelEncoder.append(le)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=666)

    # 使用 XGBoost 训练模型
    params = {
        'n_estimators': 30,
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

    _X_train, _X_val, _y_train, _y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=666)
    xgb_cls = xgb.XGBClassifier(**params, enable_categorical=True,
                                eval_metric=mean_squared_error,
                                early_stopping_rounds=3,
                                random_state=666, ).fit(_X_train, _y_train, eval_set=[(_X_val, _y_val)])
    # 预测测试集
    y_pred = np.round(xgb_cls.predict(X_test))
    y_pred_proba = xgb_cls.predict_proba(X_test).max(axis=-1)
    # y_pred_proba_1 = xgb_cls.predict_proba(X_test)[:, 1]
    y_pred_proba_train = xgb_cls.predict_proba(X_train).max(axis=-1)
    # 置信度计算
    # xgb_conf = confidence = stats.ttest_ind(y_pred_proba, 1)[1]

    print(classification_report(y_test, y_pred))

    # =========================================初始化 LIME 解释器=========================================
    list_cate_indices = [X_train.columns.get_loc(_) for _ in X_train.select_dtypes(include=['category']).columns]
    list_cate_name = list(X_train.select_dtypes(include=['category']).columns)
    dict_cate_LabelEncoder = dict(zip(list_cate_name, list_LabelEncoder))

    explainer = lime_tabular.LimeTabularExplainer(X_train.to_numpy(), feature_names=X_train.columns.to_numpy(),
                                                  categorical_features=list_cate_indices,
                                                  categorical_names=list_cate_name,
                                                  class_names=np.array(['0', '1']), discretize_continuous=True)
    # prompt结构Input + Predicted Label + Confidence + Ground Truth
    # 构造Constructed Context，基于3条训练数据
    data_train = pd.concat([X_train, y_train], axis=1).reset_index(drop=True)
    data_train['proba'] = y_pred_proba_train
    data_train['lt_0.65'] = (data_train['proba'] < 0.65) + 0
    data_test = pd.concat([X_test, y_test], axis=1).reset_index(drop=True)
    list_pred_xgb, list_pred_llm = [], []

    tmp_data_test = data_test.groupby(['fraud_reported', ]).apply(lambda x: x.sample(n=25), include_groups=True, ).reset_index(drop=True)
    # for index_test, record_test in tqdm(data_test.iterrows(), colour='green'):
    for index_test, record_test in tqdm(tmp_data_test.iterrows(), colour='green'):
        # break
        # if index_test != 299:
        #     continue
        # 构造训练数据示例
        # train_sample = data_train.sample(n=3).reset_index(drop=True)
        train_sample = data_train.groupby(['fraud_reported', 'lt_0.65']).apply(lambda x: x.sample(n=1), include_groups=False, ).reset_index()
        xgb_pred = xgb_cls.predict(train_sample[X_train.columns])
        # xgb_pred_proba = xgb_cls.predict_proba(train_sample[X_train.columns]).max(axis=-1)
        xgb_pred_proba = xgb_cls.predict_proba(train_sample[X_train.columns])
        # "policy_bind_date代表保险绑定日期；incident_date是出险日期；"
        Constructed_Context = "你是汽车保险理赔的专家，我会给你提供之前客户索赔的车险数据，希望你能帮助公司预测哪些索赔是欺诈行为。这些数据包含的字段和字段含义如下：" \
                              "\n\nage是用户年龄；months_as_customer是用户成为客户的时长，以月为单位；" \
                              "policy_state是上保险所在地区；policy_csl是组合单一限制Combined Single Limit；policy_deductable是保险扣除额；" \
                              "policy_annual_premium是每年的保费；umbrella_limit是保险责任上限；insured_zip是被保人的邮编；" \
                              "insured_sex是被保人性别：FEMALE或者MALE；insured_education_level是被保人学历；insured_occupation是被保人职业；" \
                              "insured_hobbies是被保人兴趣爱好；insured_relationship是被保人关系；capital - gains是出险的资本收益；" \
                              "capital - loss是出险的资本损失；incident_type是出险类型；collision_type是碰撞类型；" \
                              "authorities_contacted是出险时联系了当地的哪个机构；incident_state是出事所在的省份；incident_city是出事所在的城市；" \
                              "incident_hour_of_the_day是出事所在的小时（一天24小时的哪个时间）；number_of_vehicles_involved是涉及的车辆数；" \
                              "property_damage是否有财产损失；bodily_injuries是身体受伤数；witnesses是目击证人数；police_report_available是是否有警察记录；" \
                              "total_claim_amount是整体索赔金额；injury_claim是伤害索赔金额；property_claim是财产索赔金额；vehicle_claim是汽车索赔金额；" \
                              "auto_make是汽车品牌；比如Audi, BMW, Toyota, Volkswagen；auto_model是汽车型号，比如A3, X5, Camry, Passat等；" \
                              "auto_year是汽车购买的年份；incident_month是出险发生时的月份；" \
                              "fraud代表是否欺诈，1或者0。\n\n" \
                              "部分字段存在缺失值，会以问号?、nan等标记，或者直接为空值。你自行决定如何理解它们。"
        Constructed_Context += f"请你根据上述指标，进行车辆出险欺诈行为检测（属于0-1二分类任务）。我会给出几个例子，请你学习这些例子后回答。"
        # for index_train, record_train in train_sample.drop(columns=['proba', 'lt_0.65']).iterrows():

        for index_train, record_train in train_sample[X_train.columns.to_list() + ['fraud_reported']].iterrows():
            # break
            Constructed_Context += f"\n\n例{index_train + 1}：\n【输入变量】："
            for k, v in record_train.items():
                if k == 'fraud_reported':
                    continue
                Constructed_Context += f"{k}: {v}, "
            # Constructed_Context += f"\n【小模型预测标签】：{xgb_pred[index_train]}\n" \
            #                        f"\n【置信度】：{xgb_pred_proba[index_train]:.4f}\n" \
            #                        f"【真实值】：{record_train['fraud_reported']}\n"
            # Constructed_Context += f"\n【真实值】：{record_train['fraud_reported']}\n\n"
            Constructed_Context += f"\n【输出为1的概率】：{xgb_pred_proba[index_train][1]:.4f}\n" \
                                   f"【真实值】：{record_train['fraud_reported']}\n"
            Constructed_Context += f"【小模型的思考过程】：因为这条记录"

            # 加入LIME的解释
            exp = explainer.explain_instance(record_train[X_train.columns].to_numpy(), xgb_cls.predict_proba, num_features=10)
            # exp.show_in_notebook(show_table=True, show_all=False)
            # exp.as_pyplot_figure()
            # plt.show()
            for rule, LIME_score in exp.as_list():
                # break
                # 如果可以拿到labelencoder，那么就要替换等号之后的数值，否则就保存不变。
                if '>' not in rule and '<' not in rule and dict_cate_LabelEncoder.get(rule.split('=')[0], False):
                    rule = '='.join([rule.split('=')[0],
                                     str(dict_cate_LabelEncoder.get(rule.split('=')[0]).inverse_transform([int(rule.split('=')[1])]).item())])
                # 增加规则
                # Constructed_Context += f"满足该条件{rule}，该条件对分类结果的贡献度是{np.abs(LIME_score):.4f}；"
                Constructed_Context += f"满足条件{rule}，对预测有{'正' if LIME_score >= 0 else '负'}向影响，影响度约{np.abs(LIME_score):.4f}；"
                pass
            # Constructed_Context = Constructed_Context.rstrip('；')
        # Constructed_Context += f"根据上述示例，请你预测一下记录的真实值，你预测的真实值写在'【真实值】：'后。请严格遵守输出格式要求。\n"
        Constructed_Context += f"\n\n根据上述示例，结合你的计算，请你预测以下记录的真实值。你的回答以'【预测值】：'开头，然后给出你预测的真实值。" \
                               f"给出你的预测值后，你需要给出该预测的概率，写在'【预测概率】：'之后。注意预测概率必须在0~1之间，精度为四位小数。" \
                               f"最后，用简短的文字解释你的回答。" \
                               f"你必须给出你预测的真实值和置信度。" \
                               f"请严格遵守输出格式要求。\n"
        # 加入测试集数据
        record_test = pd.DataFrame([record_test.values], columns=record_test.index)
        for col in record_test.columns:
            # col='auto_make'
            # record_test.dtypes
            if record_test[col].dtypes == object:
                record_test[col] = pd.Categorical(record_test[col], categories=X_train[col].cat.categories)

        xgb_pred = xgb_cls.predict(record_test[X_train.columns])
        xgb_pred_proba = xgb_cls.predict_proba(record_test[X_train.columns])
        # list_pred_xgb.append((xgb_pred.item(), xgb_pred_proba.item()))
        list_pred_xgb.append((xgb_pred.item(), xgb_pred_proba))

        Constructed_Context += f"\n【输入变量】："
        for k, v in record_test.iloc[0].items():
            if k == 'fraud_reported':
                continue
            Constructed_Context += f"{k}: {v}, "

        Constructed_Context += f"\n" \
                               f"【输出为1的概率】：{xgb_pred_proba[0][1]:.4f}"
        # Constructed_Context += f"\n"
        if index_test == 0:
            print(Constructed_Context)
            len(Constructed_Context)

        # # 大模型预测
        # prompt = Constructed_Context
        # messages = [{"role": "system", "content": "你是车险领域的风险评估专家."}, {"role": "user", "content": prompt}]
        # text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # model_inputs = tokenizer([text], return_tensors="pt").to(device)
        # generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=32)
        # generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        # list_pred_llm.append(response)

        # # 调用官方qwen API
        # dashscope.api_key = 'sk-faebb4f64e014310ae4a3d173af9d61f'
        # prompt = Constructed_Context
        # messages = [{"role": 'system', "content": "你是车险领域的风险评估专家."}, {"role": "user", "content": prompt}]
        # responses = dashscope.Generation.call("qwen1.5-72b-chat",
        #                                        messages=messages,
        #                                        result_format='message',  # 设置输出为'message'格式
        #                                        stream=False,  # 设置输出方式为流式输出
        #                                        incremental_output=False,  # 增量式流式输出
        #                                        max_tokens=64,
        #                                        )
        # if responses.status_code == HTTPStatus.OK:
        #     print(responses.output.choices[0]['message']['content'], end='')
        # else:
        #     print(f'Request id: {responses.request_id}, Status code: {responses.status_code}, '
        #           f'error code: {responses.code}, error message: {responses.message}')
        # list_pred_llm.append(responses.output.choices[0]['message']['content'])
        pass
        # 调用MA部署的qwen72B模型.
        url = ""
        prompt = Constructed_Context
        messages = [{"role": 'system', "content": "你是车险领域的风险评估专家."}, {"role": "user", "content": prompt}]
        headers = {
            'Content-Type': 'application/json',
            'X-Apig-AppCode': ''
        }
        payload = {
            "model": "/home/mind/model/qwen1_5-72b-chat",
            "temperature": 0.,
            "max_tokens": 256,
            "messages": messages,
        }
        resp = requests.post(url, json=payload, headers=headers)
        # Print result
        if resp.status_code == HTTPStatus.OK:
            list_pred_llm.append(json.loads(resp.text)['choices'][0]['message']['content'])
            # print(json.loads(resp.text)['choices'][0]['message']['content'])
        else:
            tqdm.write('模型请求失败')
            list_pred_llm.append('模型请求失败')

    # 提取大模型预测的标签和置信度
    list_pred_llm2, list_pred_llm2_proba = [], []
    for element in list_pred_llm:
        # break
        # element = json.loads(resp.text)['choices'][0]['message']['content']
        # 提取大模型预测的标签
        match = re.search(r'【预测值】：(\d+)', element)
        if match:
            list_pred_llm2.append(int(match.group(1)))
        else:
            list_pred_llm2.append('illegal output')
            print(element)
        # 提取大模型预测的概率
        match = re.search(r'【预测概率】：([-+]?\d*\.\d+([eE][-+]?\d+)?)', element)
        if match:
            list_pred_llm2_proba.append(float(match.group(1)))
        else:
            list_pred_llm2_proba.append('illegal output')
            print(element)
        # 如果置信度区间不对,那也是不合法的输出
        if match and (float(match.group(1)) < 0 or float(match.group(1)) > 1):
            list_pred_llm2[-1] = 'illegal output'
            list_pred_llm2_proba[-1] = 'illegal output'
            print(element)

    # 规整输出,去掉illegal output的记录
    list_pred_true3, list_pred_xgb3, list_pred_llm3 = [], [], []
    list_pred_xgb3_proba_1, list_pred_llm3_proba_1 = [], []
    count = 0
    for _true, (_xgb, _xgb_proba), _llm, _llm_proba in zip(y_test.tolist(), list_pred_xgb, list_pred_llm2, list_pred_llm2_proba):
        if _llm == 'illegal output':
            count += 1
            print(f'存在{count}个illegal output')
            continue
        list_pred_true3.append(_true)
        list_pred_xgb3.append(_xgb)
        list_pred_llm3.append(_llm)
        list_pred_xgb3_proba_1.append(_xgb_proba if _xgb == 1 else 1 - _xgb_proba)
        list_pred_llm3_proba_1.append(_llm_proba if _llm == 1 else 1 - _llm_proba)

    # 输出预测结果
    for _true, _xgb, _llm in zip(y_test.tolist(), list_pred_xgb, list_pred_llm):
        if _llm == 'illegal output':
            print(f'存在个illegal output')
            continue
        print(f"【真实值】：{_true}")
        print(f"【XGBoost】预测值：{_xgb}")
        print('LLM的'+_llm)

    # ==========================================================
    # 模型评估
    # ==========================================================
    print("=" * 60)
    print("【XGBoost】分类结果评估:\n")
    res_clf = classification_report(list_pred_true3, list_pred_xgb3, digits=4, zero_division=0, )
    print(res_clf)
    print(f"【XGBoost】的lift: {lift_score(list_pred_true3, list_pred_xgb3):.4f}")
    print(f"【XGBoost】的AUC: {roc_auc_score(list_pred_true3, list_pred_xgb3_proba_1):.4f}")
    print(f"【XGBoost】的KS得分: {calculate_ks_score(list_pred_true3, list_pred_xgb3_proba_1):.4f}")
    # 使用sklearn包计算KS值
    fpr, tpr, _ = roc_curve(list_pred_true3, list_pred_xgb3_proba_1)
    ks_sklearn = np.max(np.abs(tpr - fpr))
    print(f"【XGBoost】的KS得分 (sklearn包计算): {ks_sklearn:.4f}")

    print("=" * 60)
    print("【QWen 72B】分类结果评估:\n")
    res_clf = classification_report(list_pred_true3, list_pred_llm3, digits=4, zero_division=0, )
    print(res_clf)
    print(f"【QWen 72B】的lift: {lift_score(list_pred_true3, list_pred_llm3):.4f}")
    print(f"【QWen 72B】的AUC: {roc_auc_score(list_pred_true3, list_pred_llm3_proba_1):.4f}")
    print(f"【QWen 72B】的KS得分: {calculate_ks_score(list_pred_true3, list_pred_llm3_proba_1):.4f}")
    # 使用sklearn包计算KS值
    fpr, tpr, _ = roc_curve(list_pred_true3, list_pred_llm3_proba_1)
    ks_sklearn = np.max(np.abs(tpr - fpr))
    print(f"【QWen 72B】的KS得分 (sklearn包计算): {ks_sklearn:.4f}")

    print("=" * 60)

    # 保存list_pred_llm, list_pred_true3, list_pred_xgb3, list_pred_llm3, list_pred_xgb3_proba_1, list_pred_llm3_proba_1
    with open('./results/res_SuperICL_LIME.pkl', 'wb') as f:
        dict_res = {
            'list_pred_llm': list_pred_llm,
            'list_pred_true3': list_pred_true3,
            'list_pred_xgb3': list_pred_xgb3,
            'list_pred_llm3': list_pred_llm3,
            'list_pred_xgb3_proba_1': list_pred_xgb3_proba_1,
            'list_pred_llm3_proba_1': list_pred_llm3_proba_1
        }
        pickle.dump(dict_res, f)
