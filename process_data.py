
import numpy as np
import pandas as pd
import csv
import argparse
import krippendorf
import os

import pdb

# usage python3 process_data.py -l [e/s/m/g] -c [y/n]




def main(data, lang, comp_flag):
    """
    Q1: joy_1, joy_2, power_1, power_2, surprise_1, surprise_2 (6)
    Q2: anger_1, anger_2, fear_1, fear_2, tension_1, tension_2 (6)
    Q3: bitter_1, bitter_2, sad_1, sad_2 (4)
    Q4: peace_1, peace_2, tender_1, tender_2, transcendence_1, transcendence_2 (6)
    """
    variables = ['anger', 'bitter', 'fear', 'joy', 'peace',  'power',
                 'sad', 'surprise', 'tender', 'tension', 'transc',
                 'taste', 'familiar', 'lyr_und',  'q1_a_pos_v_neg',
                 'q2_a_pos_v_neg', 'q3_a_neg_v_neg', 'q4_a_neg_v_pos']
    emotions = ['anger', 'bitter', 'fear', 'joy', 'peace',  'power',
                'sad', 'surprise', 'tender', 'tension', 'transcendence']
    # get list of songs
    dirs = os.listdir('./data_normalized')
    list_songs = [_.replace('.mp3', '') for _ in dirs]
    dict_songs_agree = {}.fromkeys(list_songs)
    dict_emo_agree = {k: [] for k in emotions}

    # agree_res = pd.DataFrame(np.zeros((len(variables), len(variables))),
    #                          columns=variables,
    #                          index=variables)
    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    data = pd.DataFrame(data_sliced[1:], columns=data_sliced[0])
    # drop duplicates in case created by csv loader
    data = data.drop_duplicates()
    # clean data
    if comp_flag:
        data = data.replace('', np.nan)
    else:
        data = data.dropna()
    # evaluate krippendorf alpha per song
    for song in list_songs:
        start = song + ':1'
        end = song + ':11'
        dict_songs_agree[song] = krippendorf.alpha(reliability_data=data.loc[:, start: end],
                                                   value_domain=[1, 2, 3, 4, 5, 6, 7])

    # mean over emotions
    for emo in emotions:
        for key in sorted(dict_songs_agree.keys()):
            if key.startswith(emo):
                dict_emo_agree[emo].append(dict_songs_agree[key])
    for key in sorted(dict_emo_agree.keys()):
        print(key, np.mean(dict_emo_agree[key]))

    # mean over quadrants
    


if __name__ == "__main__":
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
        lang = 'english'
    elif args.language == 's' or args.language == 'spanish':
        file_name = ['./results/data_spanish.csv']
        lang = 'spanish'
    elif args.language == 'm' or args.language == 'mandarin':
        file_name = ['./results/data_mandarin.csv']
        lang = '中文'
    elif args.language == 'g' or args.language == 'german':
        file_name = ['./results/data_german.csv']
        lang = 'deutsch'
    elif args.language == 'a' or args.language == 'all':
        file_name = ['./results/data_english.csv',
                     './results/data_spanish.csv',
                     './results/data_mandarin.csv',
                     './results/data_german.csv']
        lang = 'all'

    data = []
    for file in file_name:
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
                data.append(row)
    main(data, lang, comp_flag)
