"""
date: 20230922
author: Wenyu Qiu
des: 1) get the "knowledge proficiency" and "knowledge relevancy" from mode trained data 
     2) eda the trained data
"""
import os
import sys
import numpy as np
import json
from predict import get_status, get_exer_params

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, ''))


def get_params_example(epoch):
    # get_status_exer_params
    get_status(epoch)
    get_exer_params(epoch)
    return

def explore_stu_status():
    fp = "result/student_stat.txt"
    with open(fp, 'rb') as f:
        lines = f.readlines()
    stu_status = []
    kps_num = 123
    for i in range(len(lines)):
        line = lines[i]
        line_data = line.decode('gbk')
        line_data = json.loads(line_data)
        stu_status.append(line_data)
        if len(line_data) != kps_num:
            print("error No.{}".format(i))
    print("finish")

    return

def check_log_kps():
    log_fp = "data/log_data.json"
    with open(log_fp, 'rb') as f:
        log_data = json.load(f)
    
    # print(type(log_data))
    # print(len(log_data))
    # print(type(log_data[0]))
    print(log_data[0].keys())
    
    # print(log_data[0]['log_num'])
    # print(log_data[0]['logs'][0].keys())
    # print(log_data[0]['logs'][0]['knowledge_code'])

    # collect knowledge info
    kps = []
    for i in range(len(log_data)):
        for j in range(len(log_data[i]['logs'])):
            tmp_kps = log_data[i]['logs'][j]['knowledge_code']
            for kp in tmp_kps:
                if kp not in kps:
                    kps.append(kp)
    kps = sorted(kps)
    print("knowledge points num is: {}, detail is \n {}".format(len(kps), kps))
    
    return 

def check_modelling_data():
    """_summary_
    figure out whether 
    1) the exers of single stu covered all the know concpt
    2) all the kp in train and test data are the same
    3) all the exers has log in train data and test data
    4) the train set contained all the kps
    """
    log_fp = "data/log_data.json"
    with open(log_fp, 'rb') as f:
        log_data = json.load(f)
    
    # figure out whether the exers of single stu covered all the know concpt
    uncomplt_stu = []
    complt_stu = []
    omit_stu = []
    for stu in log_data:
        if stu['log_num'] < 15:
            omit_stu.append(stu['user_id'])
            continue
        covered_kps = []
        for log in stu['logs']:
            for kp in log["knowledge_code"]:
                if kp not in covered_kps:
                    covered_kps.append(kp)
                    continue
        if len(covered_kps) < 123:
            uncomplt_stu.append(stu['user_id'])
        else:
            complt_stu.append(stu['user_id'])
    del stu
    del log
    del kp

    print("-"*20, "\n In the log_data.txt:")
    print("omit_stu: {}".format(len(omit_stu)))
    print("uncomplt_kps_stu: {}".format(len(uncomplt_stu)))
    print("complt_kps_stu: {}".format(len(complt_stu)))

    # 1) figure out whether all the kp in train and test data are the same
    # 2) figure out whether all the exers has log in train data and test data
    train_dfp = "data/train_set.json"
    test_dfp = "data/test_set.json"

    with open(train_dfp, 'rb') as f:
        train_data = json.load(f)
    
    train_usrs_kps = {}  #keys are user_id
    test_usrs_kps  = {}  #keys are user_id

    train_exer_ids = []
    test_exer_ids = []

    train_kps = []
    test_kps = []

    for log in train_data:
        usr_id = str(log['user_id'])
        if usr_id not in train_usrs_kps.keys():
            train_usrs_kps[usr_id] = []
        
        for kp in log['knowledge_code']:
            if kp not in train_usrs_kps[usr_id]:
                train_usrs_kps[usr_id].append(kp)
            
            if kp not in train_kps:
                train_kps.append(kp)

        if str(log["exer_id"]) not in train_exer_ids:
            train_exer_ids.append(str(log["exer_id"]))

    del log

    with open(test_dfp, 'rb') as f:
        test_data = json.load(f)
    for stu in test_data:
        usr_id = str(stu['user_id'])
        if usr_id not in test_usrs_kps.keys():
            test_usrs_kps[usr_id] = []
        for log in stu['logs']:
            for kp in log['knowledge_code']:
                if kp not in test_usrs_kps[usr_id]:
                    test_usrs_kps[usr_id].append(kp)
                if kp not in test_kps:
                    test_kps.append(kp)
            if str(log['exer_id']) not in test_exer_ids:
                test_exer_ids.append(str(log['exer_id']))

    del stu
    del log
    del kp

    # 1) figure out whether all the kp in train and test data are the same
    same_kps_usrs = []
    diff_kps_usrs = []
    train_more = []
    test_more = []
    for usr_id in test_usrs_kps.keys():
        test_usrs_kps[usr_id] = sorted(test_usrs_kps[usr_id])
        train_usrs_kps[usr_id] = sorted(train_usrs_kps[usr_id])
        if test_usrs_kps[usr_id] == train_usrs_kps[usr_id]:
            same_kps_usrs.append(usr_id)
        else:
            if train_usrs_kps[usr_id] > test_usrs_kps[usr_id]:
                train_more.append(usr_id)
            else:
                test_more.append(usr_id)
            diff_kps_usrs.append(usr_id)
    print("\n whether all the kp in train and test data are the same for sin stu \n same_kps_usrs num: {}, \n diff_kps_usrs: {}, \n train_more num: {}, test_more num: {}".format(len(same_kps_usrs), len(diff_kps_usrs), len(train_more), len(test_more)))

    # 2) figure out whether all the exers has log in train data and test data
    total_exer_ids = list(set(train_exer_ids + test_exer_ids))
    only_train_exer_ids = []
    only_test_exer_ids = []
    for _id in total_exer_ids:
        if _id not in train_exer_ids:
            only_test_exer_ids.append(_id)
        if _id not in test_exer_ids:
            only_train_exer_ids.append(_id)
    print('\n whether all the exers has log in train data and test data \n len(train_exer_ids): {}, len(test_exer_ids): {}'.format(len(train_exer_ids), len(test_exer_ids)))
    print("total exers num: {}, \n only_train_exer_ids num: {}, only_test_exer_ids num: {}".format(len(total_exer_ids), len(only_train_exer_ids), len(only_test_exer_ids)))


    print('\n train_kps num: {}, test_kps num: {}'.format(len(train_kps), len(test_kps)))

def                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     

if __name__ == "__main__":
    # get_params_example(5)
    # explore_stu_status()
    # check_log_kps()
    check_modelling_data()