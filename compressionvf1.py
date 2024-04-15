import glob

import matplotlib.pyplot as plt
import pandas as pds
import numpy as np

MODELS = ['KNNeucli', 'Minirocket']
METRICS = ['ACCURACY', 'PRECISION', 'RECALL', 'PRECISION_MACRO', 'RECALL_MACRO', 'F1_SCORE', 'F1_SCORE_MACRO', 'F1_SCORE_WEIGHTED',
           'ROC_AUC_SCORE', 'ROC_AUC_SCORE_MACRO', 'ROC_AUC_SCORE_WEIGHTED']

paths = glob.glob("results\\ukdale/*/*/_*.json")


def split_paths(paths):
    """
    Split the paths into the different parts.
    """
    obj = {}

    for path in paths:
        path_split = path.split('\\')

        param_split = path.split('_')

        # Third split is the appliance
        if path_split[2] not in obj:
            obj[path_split[2]] = {}

            # Create one for each model
            for keys in MODELS:
                obj[path_split[2]][keys] = []

            obj[path_split[2]][param_split[-2]].append(path)
        else:
            obj[path_split[2]][param_split[-2]].append(path)

    return obj


def create_graphs(data):
    """
    Create the graphs.
    """
    for appliance in data:
        for model in data[appliance]:
            result_paths = data[appliance][model]

            df = create_dataframe(result_paths)

            group = df.groupby(['param'])[METRICS].agg(['mean'])
            print(group)

            for metric in METRICS:
                plt.figure(figsize=(6, 6))
                plt.plot(group.index, group[metric]['mean'])
                plt.ylabel(f'{metric}')
                plt.xlabel(f'% Change Compression')
                plt.xticks(group.index[::2],  rotation='vertical')
                plt.savefig(f'results/graphs/{appliance}/{model}_{metric}.png')


def create_dataframe(result_paths):
    dfs = []
    for result_path in result_paths:
        print(f'Loading up {result_path}')

        # Get the param used for this result
        param = result_path.split('_')[-1].replace('.json', '')

        # Load the data
        result = pds.read_json(result_path)
        result['param'] = param

        dfs.append(result)

    print('Combining dataframes... ')

    return pds.concat(dfs, ignore_index=True)


split = split_paths(paths)
create_graphs(split)

