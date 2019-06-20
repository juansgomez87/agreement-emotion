
import numpy as np
import pandas as pd
import csv
import argparse
import krippendorf
import os

import pdb


def print_gold_msi_data(data):
    print('GOLD-MSI scores:')
    print('F1 - Active engagement: range 9-63 > population mean 41.52')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['act_eng:1']), np.nanstd(data['act_eng:1'])))
    print('F2 - Perceptual abilities: range 9-63 > population mean 50.20')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['perc_abi:1']), np.nanstd(data['perc_abi:1'])))
    print('F3 - Musical training: range 7-49 > population mean 26.52')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['mus_trai:1']), np.nanstd(data['mus_trai:1'])))
    print('F4 - Emotions: range 6-42 > population mean 34.66')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['emo:1']), np.nanstd(data['emo:1'])))
    print('F5 - Singing abilities: range 7-49 > population mean 31.67')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['sing_abi:1']), np.nanstd(data['sing_abi:1'])))
    print('F6 - General sophistication: range 18-126 > population mean 81.58')
    print('Sample score (mean - {:.2f}, std - {:.2f})'.format(np.nanmean(data['mus_soph:1']), np.nanstd(data['mus_soph:1'])))



def main(data, comp_flag, code_lng, num_surv):
    """

    """
    # configuration flags
    pretty_print = False
    print_gold_msi = False
    # selector of subsets: 0 - select all, 1 - select positive, 2 - select negative
    # e.g. 1 - select surveys that understand lyrics, 2 - select surveys that don't understand lyrics
    sel_understood_songs = 1

    emo_enc = {1: 'anger', 2: 'bitter', 3: 'fear', 4: 'joy', 5: 'peace',  6: 'power',
               7: 'sad', 8: 'surprise', 8: 'tender', 10: 'tension', 11: 'transcendence'}
    sel_enc = {11: 'preference', 12: 'familiarity', 13: 'understanding'}
    # TODO: how to implement quadrant mapping agreement???
    quads = ['q1_a_pos_v_pos', 'q2_a_pos_v_neg',
             'q3_a_neg_v_neg', 'q4_a_neg_v_pos']
    # get list of songs
    dirs = os.listdir('./data_normalized')
    list_songs = [_.replace('.mp3', '') for _ in dirs]
    # results dictionaries
    dict_emo_mean = {k: [] for k in emo_enc.values()}
    dict_emo_std = {k: [] for k in emo_enc.values()}
    dict_emo_agree = {k: [] for k in emo_enc.values()}

    dict_quad_agree = {k: [] for k in quads}

    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    idx = pd.MultiIndex.from_arrays([code_lng])
    data = pd.DataFrame(data_sliced[1:], index=idx, columns=data_sliced[0])
    langs = np.unique(code_lng).tolist()

    # clean data from missing data
    txt_flag = 'with NaN data'
    if not comp_flag:
        data = data.dropna()
        txt_flag = 'without NaN data'
    # sample data
    if num_surv != 0:
        data = pd.concat([x.sample(n=num_surv) for x in [data.loc[_] for _ in langs]])
    # sample by preference

    # sample by familiarity

    # sample by understood_songs
    txt_understood ='understood and not understood lyrics'
    if sel_understood_songs == 1:
        txt_understood ='understood lyrics'
        for key in emo_enc.keys():
            idx = ['{}:{}'.format(_, key) for _ in list_songs]
            sel_idx = ['{}:14'.format(_) for _ in list_songs]
            pdb.set_trace()
    elif sel_understood_songs == 2:
        txt_understood ='not understood lyrics'
        

    print('**********************')
    print('{} surveys processed in {} {}'.format(data.shape[0], langs, txt_flag))
    print('Subsets: {}'.format(txt_understood))

    # mean and std across raters
    mean_raters = data.mean()
    std_raters = data.std()

    # print demographics
    print('AGE: mean - {:.2f}, std - {:.2f}'.format(mean_raters['demographics1:3'], std_raters['demographics1:3']))
    gender = [_ for _ in data['demographics2:1'].value_counts()]
    if len(gender) == 2:
        print('GENDER: men - {}, women - {}'.format(gender[0], gender[1]))
    else:
        print('GENDER: men - {}, women - {}, other - {}'.format(gender[0], gender[1], gender[2]))
    if print_gold_msi:
        print_gold_msi_data(data)
    print('**********************')
    # calculate agreement across every rating of emotion [emotion rated for all songs]
    for key in emo_enc.keys():
        idx = ['{}:{}'.format(_, key) for _ in list_songs]
        # mean across all emotions
        dict_emo_mean[emo_enc[key]] = np.mean(mean_raters[idx])
        dict_emo_std[emo_enc[key]] = np.mean(std_raters[idx])
        # agreement alpha
        dict_emo_agree[emo_enc[key]] = krippendorf.alpha(reliability_data=data[idx],
                                                         level_of_measurement='ordinal')

    # if not pretty_print:
    #     print('Mean:')
    #     for key in sorted(dict_emo_mean.keys()):
    #         print(key, dict_emo_mean[key])     
    #     print('Standard dev.:')
    #     for key in sorted(dict_emo_std.keys()):
    #         print(key, dict_emo_std[key])     
    #     print('Agreement:')
    #     for key in sorted(dict_emo_agree.keys()):
    #         print(key, dict_emo_agree[key])

    else:
        # TODO!
        pass
    # pdb.set_trace()





if __name__ == "__main__":
    # usage python3 process_data.py -l [e/s/m/g] -c [y/n] -n [integer]
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--language',
                        help='Select language to process',
                        action='store',
                        dest='language')
    parser.add_argument('-c',
                        '--complete',
                        help='Complete ratings [y] or use missing/NaN ratings [n]',
                        action='store',
                        dest='complete')
    parser.add_argument('-n',
                        '--number',
                        help='Number of surveys to process with random sampling',
                        action='store',
                        dest='number',
                        default=False)
    args = parser.parse_args()

    if args.language is None:
        print('Please select data to process!')
    if args.complete is None:
        print('Please state if you want to process all collected data or not!')

    # complete data processing
    if args.complete == 'y':
        comp_flag = True
    elif args.complete == 'n':
        comp_flag = False
    if args.language == 'e' or args.language == 'english':
        file_name = ['./results/data_english.csv']
        lang = ['english']
    elif args.language == 's' or args.language == 'spanish':
        file_name = ['./results/data_spanish.csv']
        lang = ['spanish']
    elif args.language == 'm' or args.language == 'mandarin':
        file_name = ['./results/data_mandarin.csv']
        lang = ['mandarin']
    elif args.language == 'g' or args.language == 'german':
        file_name = ['./results/data_german.csv']
        lang = ['deutsch']
    elif args.language == 'a' or args.language == 'all':
        file_name = ['./results/data_english.csv',
                     './results/data_spanish.csv',
                     './results/data_mandarin.csv',
                     './results/data_german.csv']
        lang = ['english', 'spanish', 'mandarin', 'german']

    data = []
    code_lng = []
    for num_file, file in enumerate(file_name):
        lang_name = file.split('_')[-1].replace('.csv', '')
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for num_row, row in enumerate(reader):
                # fix missing elements and convert to integers
                for idx, elem in enumerate(row):
                    try:
                        if elem == '':
                            row[idx] = np.nan
                        else:
                            row[idx] = float(elem)
                    except ValueError:
                        row[idx] = elem
                if num_row == 0 and num_file == 0:
                    # add first row with indices
                    data.append(row)
                elif num_row == 0 and num_file > 0:
                    # ignore first row of consecutive files
                    pass
                elif num_row > 0:
                    code_lng.append(lang_name)
                    data.append(row)
    main(data, comp_flag, code_lng, int(args.number))
