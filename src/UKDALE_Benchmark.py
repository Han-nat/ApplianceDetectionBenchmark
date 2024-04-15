import os, sys
import json

from sktime.classification.kernel_based import Arsenal, RocketClassifier
from sktime.classification.dictionary_based import IndividualBOSS, ContractableBOSS, BOSSEnsemble
from sktime.classification.interval_based import TimeSeriesForestClassifier, RandomIntervalSpectralEnsemble, DrCIF
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

sys.path.append(os.getcwd())
from utils._utils_ import *
from utils._utils_preprocessing_ import *

from utils.Models.InceptionTime import Inception, InceptionTime

from src.utils.downsampling.ukdale_downsampling import get_downsampled_dataset

import os.path

def launch_sktime_training(model, X_train, y_train, X_test, y_test, path_res, force_redo=True):

    if not check_file_exist(path_res+'.pt') or force_redo:
        # Equalize class for training
        X_train, y_train = RandomUnderSampler_(X_train, y_train)

        print(f'Creating model... {path_res}')
        sk_trainer = classif_trainer_sktime(model.reset(), verbose=False, save_model=False, 
                                            save_checkpoint=True, path_checkpoint=path_res) 
        print('Training')
        sk_trainer.train(X_train, y_train)
        print('Evaluating')
        metrics, predictions = sk_trainer.evaluate(X_test, y_test)
        print('Finished evaluating...')

        if isinstance(metrics, dict):
            return metrics, predictions, X_test, y_test

        return metrics.to_list(), predictions, X_test, y_test

    return


def launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test, path_res, max_epochs=100):
    
    # Equalize class for training
    X_train, y_train = RandomUnderSampler_(X_train, y_train)
    
    model_instance = model['instance']

    train_dataset = TSDataset(X_train, y_train)
    valid_dataset = TSDataset(X_valid, y_valid)
    test_dataset  = TSDataset(X_test, y_test)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=model['batch_size'], shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=True)

    model_trainer = classif_trainer(model_instance(),
                                    train_loader=train_loader, valid_loader=valid_loader,
                                    learning_rate=model['lr'], weight_decay=model['wd'],
                                    patience_es=20, patience_rlr=5,
                                    device="cuda", all_gpu=True,
                                    verbose=False, plotloss=False, 
                                    save_checkpoint=True, path_checkpoint=path_res)

    model_trainer.train(n_epochs=max_epochs)
    model_trainer.restore_best_weights()
    metrics = model_trainer.evaluate(torch.utils.data.DataLoader(test_dataset, batch_size=1))
    
    return metrics.to_list()


def UKDALE_case(chosen_clf, classifiers, list_case, list_param, path_res):

    model = classifiers[chosen_clf]
    
    for case in list_case:

        dir_res_case = create_dir(path_res+case['app']+os.sep)
        
        for param in list_param:
            try:
                results = []
                dir_res_param = create_dir(dir_res_case + param['func_name'] + os.sep
                                           + os.sep + str(param['window_size']) + os.sep)
                results_file = f'{dir_res_param}_{chosen_clf}_{param["threshold"]}.json'

                #if os.path.isfile(results_file):
                #    continue

                databuilder = UKDALEData(appliance_names=[case['app']],
                                         sampling_rate=param['sampling_rate'],
                                         window_size=param['window_size'],
                                         train_house_indicies=[1, 3, 5],
                                         test_house_indices=[2],
                                         limit_ffill=param['limit_ffill']
                                         )

                train, test = databuilder.BuildSaveNILMDataset(downsampling_func=get_downsampled_dataset,
                                                               func_name=param['func_name'],
                                                               threshold=param["threshold"])

                for seed in range(5):
                    if os.path.isfile(results_file):
                        print('skipping because results exist')
                        continue

                    np.random.seed(seed=seed)
                    np.random.shuffle(train)
                    X_train, y_train = TransformNILMDatasettoClassif(train, threshold=case['threshold'])
                    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2,
                                                                          random_state=seed)
                    X_test, y_test = TransformNILMDatasettoClassif(test, threshold=case['threshold'])

                    path_to_save = dir_res_param + chosen_clf + '_' + str(seed)

                    if chosen_clf == "Inception":
                        path_inception = create_dir(path_to_save + os.sep)

                        # ==================== Ensemble of Inception training ===================#
                        for i in range(5):
                            if not check_file_exist(path_inception + 'Inception' + str(i) + '.pt'):
                                launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                     path_inception + 'Inception' + str(i))
                        if not os.path.isfile(results_file):
                            metrics = launch_sktime_training(InceptionTime(Inception(), path_inception, 5), X_train,
                                                         y_train, X_test, y_test, path_to_save)

                    # ==================== Deep learning classifers ===================#
                    elif chosen_clf == "ResNet" or chosen_clf == "ConvNet" or chosen_clf == "ResNetAtt":
                        metrics = launch_deep_training(model, X_train, y_train, X_valid, y_valid, X_test, y_test,
                                                       path_to_save)

                    # ==================== Sktime classifers ===================#
                    else:
                        if not os.path.isfile(results_file) and len(y_test) > 0:
                            print(f'Y_test length: {len(y_test)}')
                            print(f'Running model: {chosen_clf}')
                            metrics, predictions, X_test, y_test = launch_sktime_training(model, X_train, y_train,
                                                                                          X_test, y_test, path_to_save)
                            print('Finished training and evaluating...')
                            results.append({'metrics': metrics, 'predictions': predictions.tolist(), 'y_test':
                                y_test.tolist(), 'X_test': X_test.tolist()})
                        else:
                            print('Results already exist or y_test is 0 and cannot run...')
                            continue

                print(f'Results: {results_file}')

                # Save results
                with open(results_file, "w") as file:
                    file.write(json.dumps(results))
            except Exception as e:
                print(f'Encountered Error: {e}')

    return
  

if __name__ == "__main__":

    path_res = 'new_results/' #To be fill
    
    list_case = [
                 {'app':'microwave', 'threshold': 5},
                 {'app':'kettle', 'threshold': 5}, {'app':'washing_machine', 'threshold': 5}] #
    
    # =============== Random convolution =============== #
    clf_arsenal = Arsenal(n_jobs=-1)
    clf_rocket = RocketClassifier(rocket_transform='rocket', n_jobs=-1)
    clf_minirocket = RocketClassifier(rocket_transform='minirocket', n_jobs=-1)
    
    # =============== Random interval =============== #
    clf_tsfc = TimeSeriesForestClassifier(min_interval=10, n_jobs=-1)
    clf_rise = RandomIntervalSpectralEnsemble(n_jobs=-1)
    clf_drcif = DrCIF(n_jobs=-1)
    
    # =============== Dictionnary =============== #
    clf_boss = IndividualBOSS(n_jobs=-1)
    clf_eboss = BOSSEnsemble(n_jobs=-1)
    clf_cboss = ContractableBOSS(n_jobs=-1)

    # =============== Shape distance =============== #
    clf_knne = KNeighborsTimeSeriesClassifier(algorithm='auto', distance='euclidean', n_jobs=-1)
    clf_knndtw = KNeighborsTimeSeriesClassifier(algorithm='auto', distance='dtw', n_jobs=-1)
    
    classifiers = {'Arsenal': clf_arsenal,
                   'Rocket'  : clf_rocket,
                   'Minirocket': clf_minirocket,
                   'TimeSeriesForest': clf_tsfc,
                   'Rise': clf_rise,
                   #'DrCIF': clf_drcif,
                    'BOSS': clf_boss,
                   'eBOSS': clf_eboss,
                   'cBOSS': clf_cboss,
                   #'KNNeucli' : clf_knne,
                   #'KNNdtw' : clf_knndtw,
                   #'ResNet': {'model_inst': ResNet, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   #'Inception': {'model_inst': Inception, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   #'ConvNet': {'model_inst': ConvNet, 'batch_size': 32, 'lr': 1e-3, 'wd': 0},
                   #'ResNetAtt': {'model_inst': ResNetAtt, 'batch_size': 32, 'lr': 0.0002, 'wd': 0.5}
                  }

    thresholds = [0.02, 0.04, 0.05, 0.10, 0.15, 0.20, 0.25]
    window_sizes = [48, 128, 360, 720, 1440]
    funcs = ['every_n', 'perc_change_prev_values', 'perc_change_last_transmitted', 'absolute_change_prev_values', 'absolute_change_last_transmitted']#, ]

    params = [{'sampling_rate': '1T', 'window_size': window, 'limit_ffill': 60,
               'threshold': threshold, 'func_name': func}
              for threshold in thresholds for window in window_sizes for func in funcs]

    print(params)

    # ====== List of dict parameters for resampling ===== #
    list_param = [#{'sampling_rate': '30T', 'window_size': 48,   'limit_ffill': 2},
                  #{'sampling_rate': '15T', 'window_size': 96,   'limit_ffill': 4},
                  #'sampling_rate': '10T', 'window_size': 144,  'limit_ffill': 6},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0, 'func_name': 'every_n'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .02, 'func_name': 'every_n'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .04, 'func_name': 'every_n'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .05, 'func_name': 'every_n'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.10, 'func_name': 'every_n'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.15, 'func_name': 'every_n'},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.20, 'func_name': 'every_n'},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.25, 'func_name': 'every_n'}  ,
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .02, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .04, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': .05, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.10, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.15, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.20, 'func_name': 'perc_change_prev_values'},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.25, 'func_name': 'perc_change_prev_values'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': .02,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': .04,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': .05,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.10,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.15,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.20,
                     'func_name': 'perc_change_last_transmitted'},
                    {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.25,
                     'func_name': 'perc_change_last_transmitted'}
            ]

    """
                    {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.01},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.02},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.03},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.04},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.05},
                  {'sampling_rate': '1T',  'window_size': 360, 'limit_ffill': 60, 'threshold': 0.06},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.07},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.08},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.09},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.10},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.11},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.12},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.13},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.14},
                  {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.15},
                 {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.16},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.17},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.18},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.19},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.20},
                
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.22},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.23},
                {'sampling_rate': '1T', 'window_size': 360, 'limit_ffill': 60, 'threshold': 0.24},
             
    """

    for classifier in classifiers.keys():
        UKDALE_case(classifier, classifiers, list_case, params, path_res)
        
