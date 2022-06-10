from util.Util import read_csv
import json
from tqdm import tqdm
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import spacy
from spacy.language import Language
from spacy_langdetect import LanguageDetector


# Wrapper for spacy langdetect, passed to add_pipe - has to be written like so
def get_lang_detector(nlp, name):
    return LanguageDetector()


class DataframeCleaner:

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def shape(self) -> str:
        return self.df.shape

    def save_df(self, path:str, sep = ',') -> str:
        self.df.to_csv(path, sep=sep, encoding='utf-8-sig', index=False)
        print(f"Saved df of size {self.df.shape} to path {path}")

    def n_rows(self) -> int:
        return len(self.df)


    def print_dropped(self, before: str, func_name: str = None) -> None:
        """
        Helper function to print the number of rows dropped whenever a cleanup operation is performed
        """
        after = self.n_rows()
        func = f"{func_name}:\t" if func_name else ""
        print(f"{func}Dropped {before - after} | Before: {before}, After: {after}")

    def __repr__(self) -> str:
        return f"Dataframe: {self.shape()}, cols: {self.df.columns.names()}"
    
    # -- Drop rows -- 
    def drop_outside_EU(self, continent_codes: dict) -> None:
        """
        Given contient codes in json format as seen in: http://country.io/data/, the function removes all countries that dont belong to EU
        """
        before = self.n_rows()
        filter = [continent_codes[c] == 'EU' for c in self.df['country']]
        self.df = self.df[filter]
        self.print_dropped(before, "drop_outside_EU")

    def drop_duplicates(self, col_name: str) -> None:
        """
        Drops all duplicates of ids in rows, keeping the first row
        """
        before = self.n_rows()
        self.df = self.df.drop_duplicates(subset=[col_name],keep='first')
        self.print_dropped(before, "drop_duplicates")
    
    def drop_col(self, col_name:str) -> None:
        """
        Drop column(s) by column name
        """
        self.df.drop(col_name, axis=1, inplace=True)

    def drop_na_rows(self, col_name:str) -> None:
        """
        Drop all rows that have NA values in the given column name(s)
        """
        before = self.n_rows()
        self.df = self.df.dropna(axis='rows', subset = [col_name])
        self.print_dropped(before, "drop_na_rows")

    def drop_empty_rows(self, col_name: str) -> None:
        """
        Drop all rows that have empty strings as values in a given column name
        """
        before = self.n_rows()
        self.df = self.df[self.df[col_name] != ""]
        self.print_dropped(before, "drop_empty_rows")


    def drop_long_rows(self, col_name:str) -> None:
        """
        Drop all rows where the strings at the col_name are very long (outliers)
        """
        before = self.n_rows()
        # Remove where words count is too high, count words by counting num spaces and adding 1
        word_counts = self.df[col_name].str.count(' ').add(1)
        word_limit = np.mean(word_counts) + 2*np.std(word_counts)

        self.df = self.df[word_counts <= word_limit]

        # Remove where char count is too high
        char_counts = self.df[col_name].str.len()
        char_limit = np.mean(char_counts) + 2*np.std(char_counts)

        self.df = self.df[char_counts <= char_limit]
        self.print_dropped(before, "drop_long_rows")


    def drop_non_english(self, col_name:str, threshold = 0.99) -> None:
        """
        Warning: Takes a long time to execute.
        Uses spacy's nlp language detection to drop all rows that it deems to be non english with a threshold value

        Parameters
        ----------
        col_name: str, required
            the column to translate to translate

        threshold: float, optional
            the threshold at which it is determined if the text is valid english
        """
        before = self.n_rows()

        nlp = spacy.load("en_core_web_sm")
        Language.factory("language_detector", func=get_lang_detector)
        nlp.add_pipe('language_detector', last=True)

        filter = [False for _ in range(self.n_rows())]
        for i, data in tqdm(enumerate(self.df.iterrows())):

            dict = nlp(data[1][col_name])._.language
            language, certainty = dict['language'], dict['score']
            filter[i] = language == 'en' and threshold < certainty
        
        self.df = self.df[filter]
        self.print_dropped(before, "drop_non_english")


    
    # -- String processing -- 
    def lower_text(self, col_name:str) -> None:
        self.df[col_name] = self.df[col_name].astype(str).str.lower()

    def remove_excess_spaces(self, text) -> str:
        return " ".join(text.split())

    def remove_non_text(self, col_name:str) -> None:
        # Replace everything non alphabetical with a space
        self.df[col_name] = self.df[col_name].str.replace(r'[^A-Za-z ]', ' ', regex=True)
        # Strip all excess spaces, including tab and return
        self.df[col_name] = self.df[col_name].apply(self.remove_excess_spaces)
        

    def add_no_stopwords_col(self, col_name:str) -> str:
        """
        Adds a copy of the column with the given column name. The duplicated column has had the stopwords removed
        Returns the name of the new column, which will be 'col_name'_no_stopwords
        """
        stop_words = set(stopwords.words('english'))
        new_col_name = f'{col_name}_no_stopwords'
        self.df[new_col_name] = self.df[col_name].astype(str).apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
        return new_col_name

    def add_stemmed_col(self, col_name:str) -> str:
        """
        Add a new column that is a copy of the given column, but has had all the words stemmed
        Returns the name of the new column, which will be 'col_name'_stemmed
        """
        stemmer = SnowballStemmer('english')
        new_col_name = f'{col_name}_stemmed'
        self.df[new_col_name] = self.df[col_name].apply(lambda x: ' '.join([stemmer.stem(word) for word in x.split()]))
        return new_col_name


# Demo
if __name__ == "__main__":
    df = read_csv('data/raw_data/combined.csv')
    
    # Load in which continent each country is in
    f = open('util/continent_codes.json')
    continent_codes = json.load(f)

    cleaner = DataframeCleaner(df)

    # Drop text column
    cleaner.drop_col('text')

    # Drop rows
    cleaner.drop_outside_EU(continent_codes)
    cleaner.drop_duplicates('id')
    cleaner.drop_duplicates('description')
    cleaner.drop_na_rows('description')
    cleaner.drop_long_rows('description')
    cleaner.drop_non_english('description')

    # Clean text
    cleaner.lower_text('description')
    cleaner.remove_non_text('description')
    cleaner.drop_empty_rows('description')
    no_stop_words = cleaner.add_no_stopwords_col('description')
    stemmed_text = cleaner.add_stemmed_col(no_stop_words)

    cleaner.save_df("data/processed_data/cleaned.csv")
    

    # CLEAN COMPARISON DATA AGNEWS
    # df = read_csv('data/comparison_data/agnews.csv')
    # df = df.rename(columns={'Description':'description'})
    # df = df.rename(columns={'Class Index':'Rating'})
    # df = df[['Rating',"description"]]

    # cleaner = DataframeCleaner(df)
    # cleaner.drop_duplicates('description')
    # cleaner.drop_na_rows('description')
    # cleaner.drop_long_rows('description')

    # # Clean text
    # cleaner.lower_text('description')
    # cleaner.remove_non_text('description')
    # cleaner.drop_empty_rows('description')
    # no_stop_words = cleaner.add_no_stopwords_col('description')
    # stemmed_text = cleaner.add_stemmed_col(no_stop_words)

    # cleaner.save_df("data/comparison_data/agnews_cleaned.csv")

