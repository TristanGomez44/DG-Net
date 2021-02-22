import os
from utils import get_config
import argparse
import yaml
import numpy as np
from shutil import copyfile
import sqlite3
import optuna
import gc
import glob
import torch

def run(config,trial):

    config["lr2"] = trial.suggest_float("lr2",0.001,0.003,step=0.001)
    config["lr_d"] = trial.suggest_float("lr_d",0.00005,0.00015,step=0.00005)
    config["lr_g"] = trial.suggest_float("lr_g",0.00005,0.00015,step=0.00005)
    config["batch_size"] = trial.suggest_int("batch_size",5,config["max_batch_size"],log=True)
    config["weight_decay"] = trial.suggest_float("weight_decay",0.0001,0.001,step=0.0001)
    #config["erasing_p"] = trial.suggest_float("erasing_p",0.1,0.9,step=0.2)
    #config["gan_w"] = trial.suggest_float("gan_w",0.6,1.4,step=0.2)
    #config["id_w"] = trial.suggest_float("id_w",0.6,1.4,step=0.2)
    #config["pid_w"] = trial.suggest_float("pid_w",0.6,1.4,step=0.2)
    epochLen = trial.suggest_int("epochLen",1417,1817,step=100)

    bestAcc = 0

    initEpoch = config["max_iter"]

    for epoch in range(config["epochs"]):
        config["max_iter"] += epochLen

        with open(config["config"].replace(".yaml","_trial{}.yaml".format(trial.number)),"w") as file:
            yaml.dump(config, file)

        print("config",config["config"].replace(".yaml","_trial{}.yaml".format(trial.number)),config["max_iter"])

        os.system("python train.py --config {} --name {} --resume".format(config["config"].replace(".yaml","_trial{}.yaml".format(trial.number)),config["name"]))

        iterations = config["max_iter"]

        os.system("cd reid_eval/ && python3 test_2label.py --name {} --which_epoch {} --multi".format(config["name"],iterations))
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
    parser.add_argument('--epochs', type=int, default=1)

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
