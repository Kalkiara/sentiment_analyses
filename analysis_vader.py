
import argparse
import os
import json
import re
import nltk
import glob

import pandas as pd
import numpy as np
import scipy.stats as stats

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer # (https://github.com/brunneis/vader-multi/blob/master/README.md)
nltk.download('vader_lexicon') 

def arg_inputs():
    # initilize parser
    my_parser = argparse.ArgumentParser(description="Parsing arguments for country.")

    # add arguments
    my_parser.add_argument("-c", 
                    "--country", # epochs
                    type = str,
                    required = True,
                    help = "Choose country code -- da, pl, fi, no, sv ...") 
    
                      
    # list of arguments given
    args = my_parser.parse_args()

    return args

arguments = arg_inputs()


# point to preprocessed data and out 
infile = glob.glob(os.path.join("..","preprocessed", arguments.country, "*orignal_withRT.ndjson"))
outfile = os.path.join("..","preprocessed", arguments.country + "_df.csv")


print(infile)

# read preprocessed data (tweet counts) function
def load_data(filepath: str) -> pd.DataFrame:
    """ 
    Loads json data in the specified filepath to a pandas DataFrame
    and assigns tweets to proR or proU
    """

    for i in filepath:

        r = r"_pro(.*?)_"
        string = i

        if re.findall(r, string) == ['R']:
            proR = pd.read_json(i, lines = True)
            proR = proR.assign(
                pro = lambda dataframe: 'ProR')

            print(f'[INFO] pro-russion df created  with n lines: {proR.shape[0]}')
                    
        elif re.findall(r, string) == ['U']:
            proU = pd.read_json(i, lines = True)
            proU = proU.assign(
                pro = lambda dataframe: 'ProU')

            print(f'[INFO] pro-ukranian df created  with n lines: {proU.shape[0]}')
    
    df = pd.concat([proR, proU], axis=0)
    print(f'[INFO] combined df created  with n lines: {df.shape[0]}')   

    return df


# clean data function
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Cleans pandas DataFrame:
    - removes mentions, URLs, and linebreaks
    """

    # remove mentions and URLs   
    tweets_list = df['text'].tolist() 
    tweets_list_clean = [] 

    for i in tweets_list:
        clean = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", i)
        tweets_list_clean.append(clean)

    df['text_clean'] = tweets_list_clean 

    # remove line break \n
    tweets_list = df['text_clean'].tolist() 
    tweets_list_clean = [] 

    for i in tweets_list:
        clean = re.sub("\n", "", i)
        tweets_list_clean.append(clean)

    df['text_clean'] = tweets_list_clean 

    print("[INFO] df cleaned")
    return df

# filter language function
def filter_language(df: pd.DataFrame, lang: str, filter: bool = True) -> pd.DataFrame:
    """ 
    Filters tweets based on language 
    Args:
        df: pandas DataFrame
        lang: language code, e.g. pl, en, de
        filter: boolean of whether to filter or not
    """
    if filter == True:
        df = df[df['lang'] == lang]
        print(f"[INFO] df filtered according to lang: {lang} with n lines: {df.shape[0]}")
        
    # add print for no filter
    return df

# apply sentiments where possible function
def apply_sentiments(df: pd.DataFrame) -> pd.DataFrame:
    """ 
    Applies sentiment scores for pos, neu and neg:  
    Any tweets that return errors are skipped and marked with 
    a "remove" label, then filtered out
    """
    print("[INFO:] running vader analysis...")

    sid_multi = SentimentIntensityAnalyzer()
    scores = []
    n = 0

    for i in df['text_clean']:
        
        try:
            score = sid_multi.polarity_scores(i)
            scores.append(score)
            n += 1
            
            # print to see that it is running
            if n % 10 == 0:
                print(f"[INFO:] {n} tweets of {len(df)} finished running")

        except IndexError:
            print(f"list index out of range: {df['text_clean'][n]}")
            scores.append("remove")
            n += 1

        except:
            print("Something else went wrong")
            scores.append("remove")
            n += 1


    df['tweets_sent_all'] = scores
    df = df[df['tweets_sent_all'] != "remove"]

    # positive, negatve and neutral individually
    print("[INFO:] running vader analysis for pos, neg and neu individually...")

    scores_pos = []
    scores_neg = []
    scores_neu = []
    
    n = 0

    for i in df['text_clean']:
        score_pos = sid_multi.polarity_scores(i)['pos'] 
        score_neg = sid_multi.polarity_scores(i)['neg'] 
        score_neu = sid_multi.polarity_scores(i)['neu'] 

        scores_pos.append(score_pos)
        scores_neg.append(score_neg)
        scores_neu.append(score_neu)
        n += 1

        # print to see that it is running
        if n % 10 == 0:
            print(f"[INFO:] {n} tweets of {len(df)} finished running")

    df['tweets_sent_pos'] = scores_pos
    df['tweets_sent_neg'] = scores_neg
    df['tweets_sent_neu'] = scores_neu

    return df

def main(filepath, lang):
    df = clean(load_data(filepath))
    df = filter_language(df, lang)
    df = apply_sentiments(df)
    return df


if __name__ == "__main__": 
    arguments = arg_inputs()

    df = main(infile, arguments.country) # da, pl, fi, no, sv ...
    df.to_csv(outfile)
    print('done')