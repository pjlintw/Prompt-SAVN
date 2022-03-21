"""Plotting line chart of accuracy."""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import logging
import seaborn as sns

logger = logging.getLogger(__name__)

sns.set_style("darkgrid")

def read_json(fname):
    with open(fname) as f:
        data = json.load(f)
    return data


def main():
    ### configurate here ###
    
    scores_json_file = "../low_resource.json"
    models = ["freeze_prompt-savn", "freeze_savn"]
    split_names = ["test"]
    metric_names = ["joint", "slot", "cls", "max"]
    num_epoch=5
    #  32, 64, 100, 300,
    exmaples = [ 500, 800, 1000, 3000, 5000, 8420]
    num_epoch = len(exmaples)
    log_file = "sa_train.log"
    metric_to_label = {"joint": "Joint Accuracy",
                       "slot": "Slot Accuracy",
                       "cls": "Slot Gate Accuracy",
                       "max": "Max Accuracy"}

    scores_json_file = "../low_resource.json"
    # dir_prefix = "output/{}_SA_sp_2.1_seed-{}"
    dir_prefix = "output/{}_SA_sp_2.1_seed-1017_{}"    
    ### configurate here 

    scores = read_json(scores_json_file)
    print(scores)

    # Stack, Mean and std
    x_axis = [ str(e) for e in exmaples]
    for split_name in split_names:
        print("Split name", split_name)
        for metric_name in metric_names:
            print("Metric name", metric_name)
            accs = list()

            for model_name in models:
                print("Model name", model_name)
                
                #print("before stack", scores[model_name][split_name][metric_name])
                # shape (num_dataset_size, num_epoch) == (9, 5)
                arr = np.stack(scores[model_name][split_name][metric_name])* 100
                print("1", arr)
                select_socres = arr[-6:,2]
                accs.append(arr)
                print(select_socres)
                
                #print("after stack", scores[model_name][split_name][metric_name].shape)
    
                plt.plot(x_axis, select_socres, label=model_name)
                # print(metric_name)
                # print("matrix:\n", arr)
                
                # print()
            # Find most gap
            print(accs[0]-accs[1])
            print()
            
        
            plt.xlabel("Data Size")
            plt.ylabel(metric_to_label[metric_name])
            plt.xticks(x_axis)
            plt.legend()
            fname = f"low_data_{split_name}_{metric_name}.pdf"
            plt.savefig(fname)
            print(f"Saving {fname}")
            plt.cla()
    

if __name__ == "__main__":
    main()