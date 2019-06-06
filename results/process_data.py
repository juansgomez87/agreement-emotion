import numpy as np
import pandas as pd
import csv
import argparse

import pdb

# usage python3 process_data.py -l [e/s/m/g]



def main(data):
    # remove additional empty spaces from csv load wrt length of header
    num_params = len(data[0])
    data_sliced = [_[:num_params] for _ in data]
    data = pd.DataFrame(data_sliced[1:], columns=data_sliced[0])
    pdb.set_trace()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l',
                        '--language',
                        help='Select language to process',
                        action='store',
                        dest='language')
    args = parser.parse_args()
                   
    if args.language is None:
        print('Please select data to process!')
        
    if args.language == 'e' or args.language == 'english':
        file_name = 'data_english.csv'
    elif args.language == 's' or args.language == 'spanish':
        file_name = 'data_spanish.csv'
    elif args.language == 'm' or args.language == 'mandarin':
        file_name = 'data_mandarin.csv'
    elif args.language == 'g' or args.language == 'german':
        file_name = 'data_german.csv'
       
    data = []
    with open(file_name, 'r') as f:
          reader = csv.reader(f, delimiter=',')
          for row in reader:
              data.append(row)
    main(data)
