import os
from utils import get_config
from utils import getBestTrial
import argparse
import yaml
import numpy as np
from shutil import copyfile
import sqlite3
import optuna
import gc
import glob
import torch
import sys

def run(config,trial):

    config["lr2"] = trial.suggest_float("lr2",0.0001,0.0041,step=0.001)
    config["lr_d"] = trial.suggest_float("lr_d",0.00001,0.00021,step=0.00005)
    config["lr_g"] = trial.suggest_float("lr_g",0.00001,0.00021,step=0.00005)
    config["batch_size"] = trial.suggest_int("batch_size",3,config["max_batch_size"],log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay",0.0001,0.001,step=0.0001)
    config["erasing_p"] = trial.suggest_float("erasing_p",0.4,0.6,step=0.1)
    config["gan_w"] = trial.suggest_float("gan_w",0.8,1.2,step=0.2)
    config["id_w"] = trial.suggest_float("id_w",0.8,1.2,step=0.2)
    config["pid_w"] = trial.suggest_float("pid_w",0.8,1.2,step=0.2)
    epochLen = trial.suggest_int("epochLen",1017,2217,step=100)

    if not config["distill"]:
        config["part_nb"]= trial.suggest_int("part_nb",3,15,step=2)
    else:
        config["part_nb"] = 3
        bestTrial = getBestTrial(config["exp_id"],config["distill"])

        teachConf = get_config("outputs/E0.5new_reid0.5_w30000/config_model{}_trial{}.yaml".format(config["distill"],bestTrial))
        if not "part_nb" in teachConf:
            config["part_nb_teacher"] = 3
        else:
            config["part_nb_teacher"] = ["part_nb"]

        config["bestTrialTeach"] = bestTrial

    bestAcc = 0

    initEpoch = config["max_iter"]

    if config["distill"]:
        config["high_res"] = True

    for epoch in range(config["epochs"]):
        config["max_iter"] += epochLen

        configPath = config["config"].replace(".yaml","_model{}_trial{}.yaml".format(config["model_id"],trial.number))

        with open(configPath,"w") as file:
            yaml.dump(config, file)

        print("config",configPath,config["max_iter"])

        ret = os.system("python train.py --config {} --name {} --resume --gpu_ids {}".format(configPath,config["name"],config["gpu_ids"]))
        if ret != 0:
            sys.exit(0)

        iterations = config["max_iter"]

        ret = os.system("cd reid_eval/ && python3 test_2label.py --name {} --which_epoch {} --config {} --multi".format(config["name"],iterations,configPath))
        if ret != 0:
            sys.exit(0)

        arr = np.genfromtxt("reid_eval/eval.csv",delimiter=",",dtype=str)
        rowFound = False
        i = 0
        while not rowFound:
            row = arr[i]
            if row[0] == "0.5" and row[1] == "rank1":
                acc = float(row[2])
                rowFound = True
            i += 1

        if not rowFound:
            raise ValueError("Can't find the metric in eval.csv :",arr)

        trial.report(acc,epoch)

        if acc > bestAcc:
            bestAcc = acc

            for bestWeight in glob.glob("outputs/E0.5new_reid0.5_w30000/checkpoints/*{}.pt".format(iterations)):
                fold = "../models/{}/".format(config["exp_id"])
                fileName = os.path.basename(bestWeight).replace(".pt","model{}_trial{}_best.pt".format(config["model_id"],trial.number))
                path = os.path.join(fold,fileName)
                os.system("cp {} {}".format(bestWeight,path))

        copyfile("reid_eval/eval.csv","reid_eval/eval{}.csv".format(iterations))


    config["max_iter"] = initEpoch

    os.system("rm outputs/E0.5new_reid0.5_w30000/checkpoints/*")
    os.system("cp ../DGNET/outputs/E0.5new_reid0.5_w30000/checkpoints/* outputs/E0.5new_reid0.5_w30000/checkpoints/")

    return bestAcc

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/latest.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--name', type=str, default='latest_ablation', help="outputs path")
    parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--model_id', type=str, default='default', help="model id")
    parser.add_argument('--exp_id', type=str, default='default', help="exp id")
    parser.add_argument('--max_batch_size', type=int, default=9)
    parser.add_argument('--optuna_trial_nb', type=int, default=25)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--distill', type=str,help="path to model to distill")
    parser.add_argument('--part_nb', type=int,help="Number of attention maps",default=3)

    opts = parser.parse_args()

    config = get_config(opts.config)

    config["max_iter"] = 100000

    os.system("rm outputs/E0.5new_reid0.5_w30000/checkpoints/*")
    os.system("cp ../DGNET/outputs/E0.5new_reid0.5_w30000/checkpoints/* outputs/E0.5new_reid0.5_w30000/checkpoints/")

    with open(opts.config,"w") as file:
        yaml.dump(config, file)

    if not (os.path.exists("../results/{}".format(opts.exp_id))):
        os.makedirs("../results/{}".format(opts.exp_id))
    if not (os.path.exists("../models/{}".format(opts.exp_id))):
        os.makedirs("../models/{}".format(opts.exp_id))

    study = optuna.create_study(direction="maximize",\
                                storage="sqlite:///../results/{}/{}_hypSearch.db".format(opts.exp_id,opts.model_id), \
                                study_name=opts.model_id,load_if_exists=True)

    config["epochs"] = opts.epochs
    config["name"] = opts.name
    config["config"] = opts.config
    config["max_batch_size"] = opts.max_batch_size
    config["exp_id"] = opts.exp_id
    config["model_id"] = opts.model_id
    config["distill"] = opts.distill
    config["optuna_trial_nb"] = opts.optuna_trial_nb
    config["gpu_ids"] = opts.gpu_ids

    def objective(trial):
        return run(config,trial=trial)

    con = sqlite3.connect("../results/{}/{}_hypSearch.db".format(opts.exp_id,opts.model_id))
    curr = con.cursor()

    failedTrials = 0
    for elem in curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall():
        if elem[1] is None:
            failedTrials += 1

    trialsAlreadyDone = len(curr.execute('SELECT trial_id,value FROM trials WHERE study_id == 1').fetchall())

    if trialsAlreadyDone-failedTrials < opts.optuna_trial_nb:
        print("N trials left",opts.optuna_trial_nb-trialsAlreadyDone+failedTrials)
        study.optimize(objective,n_trials=opts.optuna_trial_nb-trialsAlreadyDone+failedTrials)

    bestTrial = getBestTrial(opts.exp_id,opts.model_id,opts.optuna_trial_nb)
    configPath = "outputs/E0.5new_reid0.5_w30000/config_model{}_trial{}.yaml".format(opts.model_id,bestTrial)
    bestPaths = glob.glob("../models/{}/id_*model{}_trial{}_best.pt".format(opts.exp_id,opts.model_id,bestTrial))

    bestPaths = sorted(bestPaths,key=lambda x:int(os.path.basename(x).split("_")[1].split("model")[0]))
    bestPath = bestPaths[-1]

    iterations = int(os.path.basename(bestPath).split("_")[1].split("model")[0])

    ret = os.system("cd reid_eval/ && python3 test_2label.py --name {} --which_epoch {} --config {} --multi \
                        --att_maps --exp_id {} --model_id {} --which_trial {}".format(opts.name,iterations,configPath,opts.exp_id,opts.model_id,bestTrial))
    if ret != 0:
        sys.exit(0)
