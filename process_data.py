
import numpy as np
import pandas as pd
import csv
import argparse
import krippendorf
import os

import pdb


def main(data, comp_flag, code_lng, num_surv=None):
    """

    """
    # variables = ['anger', 'bitter', 'fear', 'joy', 'peace',  'power',
    #              'sad', 'surprise', 'tender', 'tension', 'transc',
    #              'taste', 'familiar', 'lyr_und',  'q1_a_pos_v_pos',
    #              'q2_a_pos_v_neg', 'q3_a_neg_v_neg', 'q4_a_neg_v_pos']
    # agree_res = pd.DataFrame(np.zeros((len(variables), len(variables))),
    #                          columns=variables,
    #                          index=variables)

    emotions = ['anger', 'bitter', 'fear', 'joy', 'peace',  'power',
                'sad', 'surprise', 'tender', 'tension', 'transcendence']
    quads = ['q1_a_pos_v_pos', 'q2_a_pos_v_neg',
             'q3_a_neg_v_neg', 'q4_a_neg_v_pos']
    # get list of songs
    dirs = os.listdir('./data_normalized')
    list_songs = [_.replace('.mp3', '') for _ in dirs]
    dict_songs_agree = {}.fromkeys(list_songs)
    dict_emo_agree = {k: [] for k in emotions}
    dict_quad_agree = {k: [] for k in quads}

    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    idx = pd.MultiIndex.from_arrays([code_lng])
    data = pd.DataFrame(data_sliced[1:], index=idx, columns=data_sliced[0])
    langs = np.unique(code_lng).tolist()
    # clean data from missing data
    if not comp_flag:
        data = data.dropna()
    # clean data if sample with num_samples
    if num_surv is not None:
        data = pd.concat([x.sample(n=num_surv) for x in [data.loc[_] for _ in langs]])

    # evaluate krippendorf alpha per song
    for song in list_songs:
        start = song + ':1'
        end = song + ':11'
        dict_songs_agree[song] = krippendorf.alpha(reliability_data=data.loc[:, start: end],
                                                   #value_domain=[1, 2, 3, 4, 5, 6, 7],
                                                   level_of_measurement='ordinal')

    # mean over emotions
    for emo in emotions:
        for key in sorted(dict_songs_agree.keys()):
            if key.startswith(emo):
                dict_emo_agree[emo].append(dict_songs_agree[key])
    for key in sorted(dict_emo_agree.keys()):
        print(key, np.mean(dict_emo_agree[key]))

    # mean over quadrants

    # Q1: joy_1, joy_2, power_1, power_2, surprise_1, surprise_2 (6)
    # Q2: anger_1, anger_2, fear_1, fear_2, tension_1, tension_2 (6)
    # Q3: bitter_1, bitter_2, sad_1, sad_2 (4)
    # Q4: peace_1, peace_2, tender_1, tender_2, transcendence_1, transcendence_2 (6)
    dict_quad_agree['q1_a_pos_v_pos'].append(np.mean(dict_emo_agree['joy']))
    dict_quad_agree['q1_a_pos_v_pos'].append(np.mean(dict_emo_agree['power']))
    dict_quad_agree['q1_a_pos_v_pos'].append(np.mean(dict_emo_agree['surprise']))

    dict_quad_agree['q2_a_pos_v_neg'].append(np.mean(dict_emo_agree['anger']))
    dict_quad_agree['q2_a_pos_v_neg'].append(np.mean(dict_emo_agree['fear']))
    dict_quad_agree['q2_a_pos_v_neg'].append(np.mean(dict_emo_agree['tension']))

    dict_quad_agree['q3_a_neg_v_neg'].append(np.mean(dict_emo_agree['bitter']))
    dict_quad_agree['q3_a_neg_v_neg'].append(np.mean(dict_emo_agree['sad']))

    dict_quad_agree['q4_a_neg_v_pos'].append(np.mean(dict_emo_agree['peace']))
    dict_quad_agree['q4_a_neg_v_pos'].append(np.mean(dict_emo_agree['tender']))
    dict_quad_agree['q4_a_neg_v_pos'].append(np.mean(dict_emo_agree['transcendence']))
    for key in sorted(dict_quad_agree.keys()):
        print(key, np.mean(dict_quad_agree[key]))



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
                        help='Complete data analysis or use missing ratings',
                        action='store',
                        dest='complete')
    parser.add_argument('-n',
                        '--number',
                        help='Number of surveys to process',
                        action='store',
                        dest='number')
    args = parser.parse_args()

    if args.language is None:
        print('Please select data to process!')
    if args.complete is None:
        print('Please state if you want to process all collected data or not!')
    if args.number is None:
        print('Please select number of surveys to process!')
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
        # print(lang_name)
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for num_row, row in enumerate(reader):
                # fix missing elements and convert to integers
                for idx, elem in enumerate(row):
                    try:
                        if elem == '':
                            row[idx] = np.nan
                        else:
                            row[idx] = int(elem)
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
