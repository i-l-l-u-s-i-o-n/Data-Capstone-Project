import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import string
import pickle as pkl
import pandas as pd


def do_law_of_zipf(data):
    '''
    Convert a dictionary (keys are language, and values are lists of sentences)
    into separate Pandas DataFrames for each language, and plots log scales of
    Ranks vs Frequencies, to visualize Zipf's Law.
    
    '''
    languages = list(data.keys())
    
    words_df_dict = dict()
    
    for language in languages:
        words_df_dict[language] = pd.DataFrame()

        words = []
        for sentence in data[language]:
            words.extend(sentence.split())

        words_df_dict[language]['word'] = words
        
    for language in languages:
        freqs = words_df_dict[language]['word'].value_counts().values
        ranks = range(1, len(freqs)+1)
        plt.plot(freqs, ranks, label=language)

    plt.ylabel('Frequency')
    plt.xlabel('Rank')
    plt.yscale('log')
    plt.xscale('log')
    

    plt.title('Zipf\'s Law')
    plt.legend()
    return plt