"""Extract scores from log files."""
import os
import re
import numpy as np
import json
from pprint import pprint

def find_scores_from_file(fname, metric_len):
    """Get metrics from log file.

    Args:
      fname: str, file name.
      metric_len: int, number of metric 
    """
    def decouple_score(score, metric_len, return_array=True):
        """Decouple the eval and test scores from list.

          eval_score: Get score every even steps, 0 or 2 or 4-th index
          test_score: Get score every odd steps, 1 or 3 or 5-th index
        """
        assert type(score) == list
        eval_score = score[0::2]
        test_score = score[1::2]
        # Metrics are evaluated every epoch
        # Length of list must be equivalent to number of epochs
        assert len(score) == metric_len*2
        assert len(eval_score) == metric_len
        assert len(test_score) == metric_len

        eval_score= [float(v) for v in eval_score]
        test_score= [float(v) for v in test_score]
        if return_array:
            eval_score = np.array(eval_score)
            test_score = np.array(test_score)
        return eval_score, test_score

    joint_acc, slot_acc, cls_acc, max_acc = list(), list(), list(), list()  
    data = read_file(fname)
  
    # Regex paatern
    re_pattern = r"([0-9]*[\.]+[0-9]*)"
    # Get joint accu
    joint_scores = re.findall(f"joint acc: {re_pattern}", data) 
    joint_acc_eval, joint_acc_test= decouple_score(joint_scores, metric_len=metric_len, return_array=False)

    # Get joint accu
    slot_scores = re.findall(f"slot acc: {re_pattern}", data) 
    slot_acc_eval, slot_acc_test= decouple_score(slot_scores, metric_len=metric_len, return_array=False)
    # Get joint accu
    cls_scores = re.findall(f"cls acc: {re_pattern}", data) 
    cls_acc_eval, cls_acc_test= decouple_score(cls_scores, metric_len=metric_len, return_array=False)
    # Get joint accu
    max_scores = re.findall(f"max_acc:{re_pattern}", data)
    max_acc_eval, max_acc_test= decouple_score(max_scores, metric_len=metric_len, return_array=False)

    result = ({"eval": joint_acc_eval, "test": joint_acc_test},
              {"eval": slot_acc_eval, "test": slot_acc_test},
              {"eval": cls_acc_eval, "test": cls_acc_test},
              {"eval": max_acc_eval, "test": max_acc_test})
    return result



def read_file(fname):
    with open(fname, "r") as f:
        data = f.read()
    return data


def main():
    """
    1. Read files and extract socres by regex
    2. Collect joint, slot, cls and max accuracies into score dictionary

    Exmaple of `scores`:
        {"prompt-save":
            {"eval":
                {"joint": [ [50, 52, 53, ...], []],
                 "slot": [ [92, 95], []],
                 "cls": [ ...],
                 "max": [ ...]},
             "test":
                {"joint": [ [50, 52, 53, ...], []],
                 "slot": [ [92, 95], []],
                 "cls": [ ...],
                 "max": [ ...]},
            },
        "savn":
            {"eval":
                    {"joint": [ [50, 52, 53, ...], []],
                     "slot": [ [92, 95], []],
                     "cls": [ ...],
                     "max": [ ...]},
             "test":
                {"joint": [ [50, 52, 53, ...], []],
                 "slot": [ [92, 95], []],
                 "cls": [ ...],
                 "max": [ ...]},
            }
        }

    """
    ### Configurate here ###
    json_file_name = "low_resource.json"
    num_epoch = 5
    # seeds = [ 10, 15, 32, 42, 1074]
    # seeds = [ 017]
    exmaples = [32,64,100, 300, 500, 800, 1000, 3000, 5000 , None]
    log_file = "sa_train.log"
    models = ["prompt-savn", "savn"]
    models = ["freeze_prompt-savn", "freeze_savn"]
    # dir_prefix = "output/{}_SA_sp_2.1_seed-{}"
    dir_prefix = "output/{}_SA_sp_2.1_seed-1017_{}"
    ### Configurate here ###

    # Create score dictioanry
    scores = dict()
    for model_name in models:
        scores[model_name] = dict()
        for split_name in ["eval", "test"]:
            scores[model_name][split_name] = dict()
            for metric_name in  ["joint", "slot", "cls", "max"]:
                scores[model_name][split_name][metric_name] = list()
    #print(scores)

    
    for model_name in models:
        # for seed in seeds:
        for num_example in exmaples:
            # Create file name
            fname = dir_prefix.format(model_name, num_example)
            fname = os.path.join(fname, log_file)

            if num_example == None:
                fname = "_".join(fname.split("_")[:-2])
                fname = os.path.join(fname, log_file)
            print(fname)

            # Extract scores from files
            joint_accs, slot_accs, cls_accs, max_accs = find_scores_from_file(fname, num_epoch)
            # print(joint_accs)
            # Add to evalaution
            scores[model_name]["eval"]["joint"].append(joint_accs["eval"])
            scores[model_name]["eval"]["slot"].append(slot_accs["eval"])
            scores[model_name]["eval"]["cls"].append(cls_accs["eval"])
            scores[model_name]["eval"]["max"].append(max_accs["eval"])
            # Add to test
            scores[model_name]["test"]["joint"].append(joint_accs["test"])
            scores[model_name]["test"]["slot"].append(slot_accs["test"])
            scores[model_name]["test"]["cls"].append(cls_accs["test"])
            scores[model_name]["test"]["max"].append(max_accs["test"])
        #     break
        # break
    # print("scores", scores)
    pprint(scores)
    with open(json_file_name, 'w') as fp:
        json.dump(scores, fp)

    



if __name__ == "__main__":
    main()
