
import numpy as np
import pandas as pd
import csv
import argparse
import krippendorf
import os
import sys

import pdb


def print_gold_msi_data(data):
    """
    """
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


def print_results(mean, std, agree, pretty_print):
    if not pretty_print:
        print('Mean:')
        for key in sorted(mean.keys()):
            # print(key, dict_emo_mean[key])
            print('{:.3f}'.format(mean[key]))    
        print('Standard dev.:')
        for key in sorted(std.keys()):
            # print(key, dict_emo_std[key])  
            print('{:.3f}'.format(std[key]))
        print('Agreement:')
        for key in sorted(agree.keys()):
            # print(key, dict_emo_agree[key])
            print('{:.3f}'.format(agree[key]))
    else:
        # TODO: implement pretty print for latex
        pass


def filter_samples(data, emo_enc, sel_enc, list_songs, idx_smp, sel):
    """
    """
    cnt = 0
    if sel == 0:
        txt_to_print = '{} (all - positive (>3) and negative(<3))\n'.format(sel_enc[idx_smp])
    elif sel == 1:
        txt_to_print = '{} (positive (>3))\n'.format(sel_enc[idx_smp])
        for key in emo_enc.keys():
            idx = ['{}:{}'.format(_, key) for _ in list_songs]
            sel_idx = ['{}:{}'.format(_, str(idx_smp)) for _ in list_songs]
            for idx_rat, idx_sel in zip(idx, sel_idx):
                # set values less than 3 to nan
                data[idx_rat].iloc[np.where(data[idx_sel] < 3)[0]] = np.nan
                # count values higher than 3
                cnt += np.sum(data[idx_sel] > 3)
    elif sel == 2:
        txt_to_print = '{} (negative (<3))\n'.format(sel_enc[idx_smp])
        for key in emo_enc.keys():
            idx = ['{}:{}'.format(_, key) for _ in list_songs]
            sel_idx = ['{}:{}'.format(_, str(idx_smp)) for _ in list_songs]
            for idx_rat, idx_sel in zip(idx, sel_idx):
                # set values higher than 3 to nan
                data[idx_rat].iloc[np.where(data[idx_sel] > 3)[0]] = np.nan
                # count values less than 3
                cnt += np.sum(data[idx_sel] < 3)
    return data, txt_to_print, cnt

def select_filter(filter):
    # selector of subsets: 0 - select all, 1 - select positive, 2 - select negative
    # e.g. 1 - select surveys that understand lyrics, 2 - select surveys that don't understand lyrics
    sel_preferred_songs = 0
    sel_familiar_songs = 0
    sel_understood_songs = 0
    if filter == 'p1':
        sel_preferred_songs = 1
    elif filter == 'p2':
        sel_preferred_songs = 2
    elif filter == 'f1':
        sel_familiar_songs = 1
    elif filter == 'f2':
        sel_familiar_songs = 2
    elif filter == 'u1':
        sel_understood_songs = 1
    elif filter == 'u2':
        sel_understood_songs = 2
    return sel_preferred_songs, sel_familiar_songs, sel_understood_songs


def main(data, comp_flag, rem_flag, quad_flag, code_lng, num_surv, filter, lang_filter):
    """

    """
    # configuration flags
    pretty_print = False
    print_gold_msi = False
    sel_preferred_songs, sel_familiar_songs, sel_understood_songs = select_filter(filter)

    emo_enc = {1: 'anger', 2: 'bitter', 3: 'fear', 4: 'joy', 5: 'peace',  6: 'power',
               7: 'sad', 8: 'surprise', 9: 'tender', 10: 'tension', 11: 'transcendence'}
    sel_enc = {12: 'preference', 13: 'familiarity', 14: 'understanding'}
    quad_enc = {'q1_a_pos_v_pos': ['joy', 'power', 'surprise'],
                'q2_a_pos_v_neg': ['anger', 'fear', 'tension'],
                'q3_a_neg_v_neg': ['bitter', 'sad'],
                'q4_a_neg_v_pos': ['peace', 'tender', 'transcendence']}
    # get list of songs
    dirs = os.listdir('./data_normalized')
    list_songs = [_.replace('.mp3', '') for _ in dirs]

    list_inst = ['anger_2', 'fear_1']
    list_spa = ['peace_2', 'sad_2', 'tender_2']
    list_non_eng = list_spa + list_inst
    list_eng = [_ for _ in list_songs if _ not in list_non_eng]
    # select songs depending on lyrics to process
    if lang_filter == 'all':
        pass
        txt_lyrics = 'with lyrics in all languages'
    elif lang_filter == 'inst':
        list_songs = list_inst
        txt_lyrics = 'with instrumental music'
    elif lang_filter == 'eng':
        list_songs = list_eng
        txt_lyrics = 'with lyrics in english'
    elif lang_filter == 'spa':
        list_songs = list_spa
        txt_lyrics = 'with lyrics in spanish'

    # results dictionaries for emotions
    dict_emo_mean = {k: [] for k in emo_enc.values()}
    dict_emo_std = {k: [] for k in emo_enc.values()}
    dict_emo_agree = {k: [] for k in emo_enc.values()}
    # results dictionaries for quadrants
    dict_quad_mean = {k: [] for k in quad_enc.keys()}
    dict_quad_std = {k: [] for k in quad_enc.keys()}
    dict_quad_agree = {k: [] for k in quad_enc.keys()}

    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    idx = pd.MultiIndex.from_arrays([code_lng])
    data = pd.DataFrame(data_sliced[1:], index=idx, columns=data_sliced[0])
    langs = np.unique(code_lng).tolist()

    # clean data from missing data
    txt_flag = 'with NaN data'
    if not comp_flag and not rem_flag:
        data = data.dropna()
        txt_flag = 'without NaN data'
    elif not comp_flag and rem_flag:
        data = data.dropna(thresh=205)
        txt_flag = 'without NaN data (if all ratings were 3 - thresh 210)'
    # sample data
    if num_surv != 0:
        data = pd.concat([x.sample(n=num_surv) for x in [data.loc[_] for _ in langs]])

    # sample by preference
    idx_smp = 12
    data, txt_pref, cnt_pref = filter_samples(data, emo_enc, sel_enc, list_songs, idx_smp, sel_preferred_songs)
    # sample by familiarity
    idx_smp = 13
    data, txt_fam, cnt_fam = filter_samples(data, emo_enc, sel_enc, list_songs, idx_smp, sel_familiar_songs)
    # sample by understood_songs
    idx_smp = 14
    data, txt_und, cnt_und = filter_samples(data, emo_enc, sel_enc, list_songs, idx_smp, sel_understood_songs)
    txt_to_print = txt_pref + txt_fam + txt_und
    tot_cnt = cnt_pref + cnt_fam + cnt_und
    tot_rat = data.shape[0] * len(emo_enc) * len(list_songs)

    if tot_cnt == 0:
        tot_cnt = tot_rat
    rate_cnt = tot_cnt / tot_rat
    print('**********************')
    print('{} surveys processed in {} {} {}'.format(data.shape[0], langs, txt_flag, txt_lyrics))
    print('Subsets:\n{}'.format(txt_to_print))
    print('{}/{} ratings ({})'.format(tot_cnt, tot_rat, rate_cnt))

    # mean and std across raters
    mean_raters = data.mean()
    std_raters = data.std()

    
    # print demographics
    # calculate demographics after filters
    print('AGE: mean - {:.2f}, std - {:.2f}'.format(mean_raters['demographics1:3'], std_raters['demographics1:3']))
    gender = [_ for _ in data['demographics2:1'].value_counts()]
    if len(gender) == 2:
        print('GENDER: men - {}, women - {}'.format(gender[0], gender[1]))
    else:
        print('GENDER: men - {}, women - {}, other - {}'.format(gender[0], gender[1], gender[2]))
    if print_gold_msi:
        print_gold_msi_data(data)
    print('**********************')
    # calculate agreement across every rating of emotion for each quad [2 or 3 emotions per quad]
    if quad_flag:
        for key, list_emo in quad_enc.items():
            num_emo = [[key for key, val in emo_enc.items() if val == _] for _ in list_emo]
            num_emo = [y for x in num_emo for y in x]
            idx = [['{}:{}'.format(song, _) for song in list_songs] for _ in num_emo]
            idx = [y for x in idx for y in x]
            # mean across all emotions
            dict_quad_mean[key] = np.mean(mean_raters[idx])
            dict_quad_std[key] = np.mean(std_raters[idx])
            # agreement alpha
            dict_quad_agree[key] = krippendorf.alpha(reliability_data=data[idx],
                                                     level_of_measurement='ordinal')
            # pdb.set_trace()
        print_results(dict_quad_mean, dict_quad_std, dict_quad_agree, pretty_print)

    # calculate agreement across every rating of emotion [emotion rated for all songs]
    else:
        for key in emo_enc.keys():
            idx = ['{}:{}'.format(_, key) for _ in list_songs]
            # mean across all emotions
            dict_emo_mean[emo_enc[key]] = np.mean(mean_raters[idx])
            dict_emo_std[emo_enc[key]] = np.mean(std_raters[idx])
            # agreement alpha
            dict_emo_agree[emo_enc[key]] = krippendorf.alpha(reliability_data=data[idx],
                                                             level_of_measurement='ordinal')
        print_results(dict_emo_mean, dict_emo_std, dict_emo_agree, pretty_print)






if __name__ == "__main__":
    # usage python3 process_data.py -l [e/s/m/g] -c [y/n] -r [y/n] -n [integer] -f [p1,p2,f1,f2,u1,u2]
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--language',
                        help='Select language to process',
                        required=True,
                        action='store')
    parser.add_argument('-c',
                        '--complete',
                        help='Complete ratings [y] or drop missing/NaN ratings [n]',
                        required=True,
                        action='store')
    parser.add_argument('-r',
                        '--remove',
                        help='Keep neutral ratings [y] or not [n]',
                        required=True,
                        action='store')
    parser.add_argument('-q',
                        '--quadrant',
                        help='Process by quadrants [y] or by emotions [n]',
                        required=True,
                        action='store')
    parser.add_argument('-lf',
                        '--lang_filter',
                        help='Process lyrics for all songs [all], instrumental [inst], english [eng], spanish [spa]',
                        required=True,
                        action='store')
    parser.add_argument('-n',
                        '--number',
                        help='Number of surveys to process with random sampling',
                        action='store')
    parser.add_argument('-f',
                        '--filter',
                        help='Select filter for data [preference, familiarity, understanding, instrumental, spanish, english]',
                        action='store')
    args = parser.parse_args()

    if args.language is None:
        print('Please select data to process!')
        sys.exit(0)
    if args.complete is None:
        print('Please state if you want to process all collected data or not!')
        sys.exit(0)
    if args.remove is None:
        print('Please state if you want to keep neutral ratings or not!')
        sys.exit(0)
    if args.quadrant is None:
        print('Please state if process by quadrants or by emotions!')
        sys.exit(0)
    if args.filter != 'f1' and args.filter != 'f2' and args.filter != 'p1' and args.filter != 'p2' and args.filter != 'u1' and args.filter != 'u2' and args.filter is not None:
        print('Please choose a valid filter!')
        sys.exit(0)
    if args.lang_filter != 'all' and args.lang_filter != 'inst' and args.lang_filter != 'eng' and args.lang_filter != 'spa' and args.lang_filter is not None:
        print('Please choose a valid lyrics filter!')
        sys.exit(0)   


    # num of surveys to sample
    if args.number is None:
        num_to_proc = 0
    else:
        num_to_proc = int(args.number)

    # complete data processing
    if args.complete == 'y':
        comp_flag = True
    elif args.complete == 'n':
        comp_flag = False

    # quadrants data processing
    if args.quadrant == 'y':
        quad_flag = True
    elif args.quadrant == 'n':
        quad_flag = False

    # neutral ratings processing
    if args.remove == 'y':
        rem_tx = '.no.neu'
        rem_flag = True
    elif args.remove == 'n':
        rem_tx = ''
        rem_flag = False

    # file selection
    if args.language == 'e' or args.language == 'english':
        file_name = ['./results/data_english{}.csv'.format(rem_tx)]
        # nat_sel = int(input('Please select all english surveys [1] or only native [2]'))
        # pdb.set_trace()
        lang = ['english']
    elif args.language == 's' or args.language == 'spanish':
        file_name = ['./results/data_spanish{}.csv'.format(rem_tx)]
        lang = ['spanish']
    elif args.language == 'm' or args.language == 'mandarin':
        file_name = ['./results/data_mandarin{}.csv'.format(rem_tx)]
        lang = ['mandarin']
    elif args.language == 'g' or args.language == 'german':
        file_name = ['./results/data_german{}.csv'.format(rem_tx)]
        lang = ['deutsch']
    elif args.language == 'a' or args.language == 'all':
        file_name = ['./results/data_english{}.csv'.format(rem_tx),
                     './results/data_spanish{}.csv'.format(rem_tx),
                     './results/data_mandarin{}.csv'.format(rem_tx),
                     './results/data_german{}.csv'.format(rem_tx)]
        lang = ['english', 'spanish', 'mandarin', 'german']

    data = []
    code_lng = []
    for num_file, file in enumerate(file_name):
        lang_name = file.split('_')[-1].replace('{}.csv'.format(rem_tx), '')
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

    main(data, comp_flag, rem_flag, quad_flag, code_lng, num_to_proc, args.filter, args.lang_filter)
