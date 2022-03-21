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
    scores_json_file = "../five_seeds_performance.json"
    models = ["prompt-savn", "savn"]
    split_names = ["eval", "test"]
    metric_names = ["joint", "slot", "cls", "max"]
    num_epoch=5

    metric_to_label = {"joint": "Joint Accuracy",
                       "slot": "Slot Accuracy",
                       "cls": "Slot Gate Accuracy",
                       "max": "Max Accuracy"}
    ### configurate here 

    
    scores = read_json(scores_json_file)
    print(scores)

    # Stack, Mean and std
    x_axis = np.arange(num_epoch) + 1
    for split_name in split_names:
        print("Split name", split_name)
        for metric_name in metric_names:
            print("Metric name", metric_name)
            for model_name in models:
                print("Model name", model_name)
                
                #print("before stack", scores[model_name][split_name][metric_name])
                arr = np.stack(scores[model_name][split_name][metric_name])
                #print("after stack", scores[model_name][split_name][metric_name].shape)
    
                avg_acc = arr.mean(axis=0) * 100
                std_acc = arr.std(axis=0) * 100
                print(metric_name)
                print("matrix:\n", arr)
                print("avergae:\n", avg_acc)
                print("std\n", std_acc)
                print()

                plt.plot(x_axis, avg_acc, label=model_name)
                plt.fill_between(x_axis, avg_acc-std_acc, avg_acc+std_acc, alpha=0.2)
        
            plt.xlabel("Epochs")
            plt.ylabel(metric_to_label[metric_name])
            plt.xticks(x_axis)
            #plt.grid()
            plt.legend()
            fname = f"{split_name}_{metric_name}.pdf"
            plt.savefig(fname)
            print(f"Saving {fname}")
            plt.cla()
    

if __name__ == "__main__":
    main()