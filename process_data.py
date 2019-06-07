import numpy as np
import pandas as pd
import csv
import argparse

import pdb

# usage python3 process_data.py -l [e/s/m/g] -c [y/n]

def clean_incomplete_data(data):
    """ This function deletes incomplete surveys
    """
    incomp_surv = []
    comp_surv = []
    for idx, row in data.iterrows():
        empty_cnt = len([_ for _ in row if _ == ''])
        if empty_cnt == 0:
            comp_surv.append(idx)
        else:
            incomp_surv.append(idx)
    print('Complete surveys:', len(comp_surv))
    print('Incomplete surveys:', len(incomp_surv))
    data = data.iloc[comp_surv]
    return data

def main(data, lang, comp_flag):
    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    data = pd.DataFrame(data_sliced[1:], columns=data_sliced[0])
    # clean data
    if not comp_flag:
        data = clean_incomplete_data(data)
    pdb.set_trace()



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
        file_name = './results/data_english.csv'
        lang = 'english'
    elif args.language == 's' or args.language == 'spanish':
        file_name = './results/data_spanish.csv'
        lang = 'spanish'
    elif args.language == 'm' or args.language == 'mandarin':
        file_name = './results/data_mandarin.csv'
        lang = '中文'
    elif args.language == 'g' or args.language == 'german':
        file_name = './results/data_german.csv'
        lang = 'deutsch'
       
    data = []
    with open(file_name, 'r') as f:
          reader = csv.reader(f, delimiter=',')
          for row in reader:
              data.append(row)
    main(data, lang, comp_flag)
