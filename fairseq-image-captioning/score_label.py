import os
import re
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score,f1_score

def main(reference_caption_file: str, system_caption_file: str):
    ref = pd.read_csv(reference_caption_file)
    pred = pd.read_csv(system_caption_file)

    acc = 0.0

    for i in range(len(pred)):
        for j in range(len(ref)):
            if str(pred.iloc[i,1]).rstrip() + '.tif' == ref.iloc[j,0]:
                preds = np.array(pred.iloc[i,2:],dtype=np.int32)
                refs = np.array(ref.iloc[j,1:],dtype=np.int32)
                acc += f1_score(refs,preds)

    total_acc = acc/len(pred)

    print('\nScores: ', total_acc)

def cli_main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)
    main(args.reference_captions, args.system_captions)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference-captions', type=lambda x: is_valid_file(parser, x, ['csv']), required=True)
    parser.add_argument('--system-captions', type=lambda x: is_valid_file(parser, x, ['csv']), required=True)
    return parser


def is_valid_file(parser, arg, file_types):
    ext = re.sub(r'^\.', '', os.path.splitext(arg)[1])

    if not os.path.exists(arg):
        parser.error('File not found: "{}"'.format(arg))
    elif ext not in file_types:
        parser.error('Invalid "{}" provided. Only files of type {} are allowed'.format(arg, file_types))
    else:
        return arg


if __name__ == '__main__':
    cli_main()