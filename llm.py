#!/usr/bin/env python
# coding: utf-8
# SuperICL预测分类任务

import pandas as pd
import pynvml
import torch
import datasets
import transformers
import random
import time
import numpy as np
import xgboost as xgb

from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm, trange
import argparse
from sklearn.metrics import matthews_corrcoef, accuracy_score, confusion_matrix, classification_report

from utils import gpt3_complete
from templates import get_input_template, get_plugin_template

from util.commonFunctions import seed_everything

# 下载开源大模型
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-110B-Chat-GPTQ-Int4", cache_dir='/data1/xuyuquan/hf_model', )
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-110B-Chat-GPTQ-Int4", cache_dir='/data1/xuyuquan/hf_model', resume_download=True)


def convert_label(label, label_list):
    if label.startswith("LABEL_"):
        return label_list[int(label.split("_")[-1])]
    else:
        return label.lower()


if __name__ == "__main__":
    # =========================================
    # 超参数设置开始
    # =========================================
    # os.chdir('./insurance')
    seed_everything(seed=666)
    num_class = 2
    num_epochs = 30
    batch_size = 128
    learning_rate = .001
    save_model = False
    load_model = False

    # 检查CUDA是否可用，并据此设置设备
    pynvml.nvmlInit()
    tmp = [pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(_)).used for _ in range(torch.cuda.device_count())]
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
    df['fraud_reported_01'] = df['fraud_reported'].map({'N': 0, 'Y': 1})
    # 删除不需要的列
    df.drop(columns='_c39', inplace=True)
    # 将所有对象类型的列转换为类别类型
    for col in df.columns:
        if df[col].dtype == 'object' and col != 'fraud_reported':
            df[col] = df[col].astype('category')

    # ==========================2. 加载训练好的小模型==========================
    # 定义特征 X 和目标变量 y
    # X = df.drop(columns=['fraud_reported', 'policy_bind_date', ])
    X = df.drop(columns=['fraud_reported', ])
    y = df['fraud_reported']

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
    xgb_cls = xgb.XGBClassifier(**params, enable_categorical=True, random_state=666, ).fit(X_train, y_train)
    # 预测测试集
    y_pred_01 = np.round(xgb_cls.predict(X_test))
    y_pred = xgb_cls.predict(X_test)
    pred_xgb = y_pred_01

    # ==========================3. 基于小模型和输入数据，构造input模板==========================
    # prompt结构Input + Predicted Label + Confidence + Ground Truth
    # 构造Constructed Context，基于3条训练数据
    train_sample = pd.concat([X_train, y_train], axis=-1).sample(n=3)
    Constructed_Context = f'请你根据一些输入指标，进行车辆出险欺诈检测。我会给出几个例子，请你学习这些例子后回答。\n\n'
    train_sample = df.sample(n=3)
    'items', 'iterrows', 'itertuples',
    for index, (_, record) in enumerate(train_sample.iterrows()):
        Constructed_Context += f'例{index + 1}：\n【输入变量】：\n'
        for k, v in record.items():
            Constructed_Context += f'{k}: {v}, '
        
        
        Constructed_Context += f'【小模型预测标签】：{}\n' \
                               f'【置信度】：{}\n' \
                               f'【真实值】：{}\n'


    for i1 in [1,2,3]:
        for i2 in 1:
            pass
        Constructed_Context += f'例{}：\n' \
                               f'【输入变量】：\n' \
                               f'{}:{}' \
                               f'【小模型预测标签】：{}\n' \
                               f'【置信度】：{}\n' \
                               f'【真实值】：{}\n'
    Constructed_Context+=f'请你预测该条记录是否是'

    # ==========================4. 调用大模型给出输出==========================

    # ==========================5. 评估模型效果==========================
    # 计算评估指标
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("【XGBoost】分类结果矩阵:")
    print(conf_matrix)
    print("=" * 60)
    res_clf = classification_report(y_test, y_pred, digits=4, zero_division=0, )
    print(res_clf)
    print("=" * 60)

    # ==========================Appendix. 其它==========================
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="RoBERTa-Large",
        help="Name of model for prompts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnli-m", "mnli-mm", "sst2", "qnli", "mrpc", "qqp", "cola", "rte"],
        help="Dataset to test on",
    )
    parser.add_argument(
        "--num_examples", type=int, default=32, help="Number of in-context examples"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--run_icl", action="store_true", default=True, help="Run ICL baseline"
    )
    parser.add_argument(
        "--run_plugin_model",
        action="store_true",
        default=True,
        help="Run plugin model baseline",
    )
    parser.add_argument(
        "--run_supericl", action="store_true", default=True, help="Run SuperICL"
    )
    parser.add_argument(
        "--sleep_time", type=float, default=0.5, help="Sleep time between GPT API calls"
    )
    parser.add_argument(
        "--explanation", action="store_true", default=False, help="Run with explanation"
    )
    args = parser.parse_args()

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    plugin_model = transformers.pipeline("text-classification", model=args.model_path)
    print(f"Loaded model {args.model_path} with name {args.model_name}")
    print(f"Testing on dataset: {args.dataset}")

    dataset_name = args.dataset.split("-")[0]
    dataset = datasets.load_dataset("glue", dataset_name)
    label_list = dataset["train"].features["label"].names

    train = dataset["train"].shuffle().select(range(args.num_examples))
    test = (
        dataset["validation"]
        if not args.dataset.startswith("mnli")
        else dataset[
            "validation" + {"m": "_matched", "mm": "_mismatched"}[args.dataset[-1]]
        ]
    )

    if args.run_icl:
        in_context_prompt = ""
        for example in train:
            in_context_prompt += f"{get_input_template(example, dataset_name)}\nLabel: {label_list[example['label']]}\n\n"

        icl_predictions = []
        icl_ground_truth = []
        for example in tqdm(test):
            valid_prompt = (
                in_context_prompt
                + f"{get_input_template(example, dataset_name)}\nLabel: "
            )
            response = gpt3_complete(
                engine="text-davinci-003",
                prompt=valid_prompt,
                temperature=1,
                max_tokens=10,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                stop=None,
            )
            time.sleep(args.sleep_time)
            icl_predictions.append(response["choices"][0]["text"].strip())
            icl_ground_truth.append(label_list[example["label"]])

        if dataset_name == "cola":
            print(
                f"ICL Matthews Corr: {matthews_corrcoef(icl_predictions, icl_ground_truth)}"
            )
        else:
            print(f"ICL Accuracy: {accuracy_score(icl_predictions, icl_ground_truth)}")

    if args.run_plugin_model:
        plugin_model_predictions = []
        plugin_model_ground_truth = []
        for example in tqdm(test):
            plugin_model_label = convert_label(
                plugin_model(get_plugin_template(example, dataset_name))[0]["label"],
                label_list,
            )
            plugin_model_predictions.append(plugin_model_label)
            plugin_model_ground_truth.append(label_list[example["label"]])

        if dataset_name == "cola":
            print(
                f"Plugin Model Matthews Corr: {matthews_corrcoef(plugin_model_predictions, plugin_model_ground_truth)}"
            )
        else:
            print(
                f"Plugin Model Accuracy: {accuracy_score(plugin_model_predictions, plugin_model_ground_truth)}"
            )

    if args.run_supericl:
        in_context_supericl_prompt = ""
        for example in train:
            plugin_input = get_plugin_template(example, dataset_name)
            plugin_model_result = plugin_model(plugin_input)[0]
            plugin_model_label = convert_label(plugin_model_result["label"], label_list)
            plugin_model_confidence = round(plugin_model_result["score"], 2)
            in_context_supericl_prompt += f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: {label_list[example['label']]}\n\n"

        supericl_predictions = []
        supericl_ground_truth = []
        for example in tqdm(test):
            plugin_input = get_plugin_template(example, dataset_name)
            plugin_model_result = plugin_model(plugin_input)[0]
            plugin_model_label = convert_label(plugin_model_result["label"], label_list)
            plugin_model_confidence = round(plugin_model_result["score"], 2)
            valid_prompt = f"{get_input_template(example, dataset_name)}\n{args.model_name} Prediction: {plugin_model_label} (Confidence: {plugin_model_confidence})\nLabel: "
            response = gpt3_complete(
                engine="text-davinci-003",
                prompt=in_context_supericl_prompt + valid_prompt,
                temperature=1,
                max_tokens=10,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=0,
                best_of=1,
                stop=None,
            )
            time.sleep(args.sleep_time)
            supericl_prediction = response["choices"][0]["text"].strip()
            supericl_ground_label = label_list[example["label"]]

            supericl_predictions.append(supericl_prediction)
            supericl_ground_truth.append(supericl_ground_label)

            if args.explanation and supericl_prediction != plugin_model_label:
                explain_prompt = (
                    in_context_supericl_prompt
                    + valid_prompt
                    + "\nExplanation for overriding the prediction:"
                )
                response = gpt3_complete(
                    engine="text-davinci-003",
                    prompt=explain_prompt,
                    temperature=1,
                    max_tokens=100,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    best_of=1,
                    stop=None,
                )
                print(f"\n{valid_prompt + supericl_prediction}")
                print(f"Explanation: {response['choices'][0]['text'].strip()}\n")

        if dataset_name == "cola":
            print(
                f"SuperICL Matthews Corr: {matthews_corrcoef(supericl_predictions, supericl_ground_truth)}"
            )
        else:
            print(
                f"SuperICL Accuracy: {accuracy_score(supericl_predictions, supericl_ground_truth)}"
            )
