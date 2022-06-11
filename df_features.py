import pandas as pd
import numpy as np
from tqdm import tqdm

from util.Util import read_csv
from copy import copy
from itertools import zip_longest

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rake_nltk import Rake # for keyword extraction

class DataFrameFeatures:

    def __init__(self, df: pd.DataFrame, vectorizer: TfidfVectorizer, glove_embeddings: dict = None, main_col: str = "description", verbose: bool = True) -> None: # description_no_stopwords_stemmed
        """
        A pd.DataFrame wrapper to perform to calculate feature vectors etc. 

        Parameters
        ----------
        df: pd.DataFrame, required
            the dataframe to build the features from, either the large dataframe or a smaller training df.
        
        vectorizer: TfidfVectorizer, required
            a tf-idf vectorizer from sklearn, used for word embeddings, 
            can be on the standard format: TfidfVectorizer(max_df=0.7,max_features=100)

        glove_embeddings: dict, optional
            a gloVe dataset converted to a dataframe using pandas, then to a dict. 
            Pretrained GloVe datasets can be found at https://nlp.stanford.edu/projects/glove/, 
            and read with pandas using pd.read_csv('glove.42B.300d.txt', sep=" ", quoting=3, header=None, 
            index_col=0).
            If not provided no glove embeddings will be calculated, useful for the entire df.

        main_col: str, optional
            the name of the column where the text which will be used to calculate the features is.

        """
        if verbose:
            print("Initializing DataFrameFeatures object")


        self.df = df
        self.main_col = main_col

        # Fit the tf-idf vectorizer initially to avoid waittime when used later
        self.vectorizer = vectorizer
        self.fitted_vectorizer = vectorizer.transform(self.documents(main_col))


        self.avg_char_count = sum([len(x) for x in self.documents(main_col)]) / self.n_rows()
        self.avg_word_count = sum([len(x.split()) for x in self.documents(main_col)]) / self.n_rows()


        # Read in the pretrained glove embeddings
        self.glove = glove_embeddings

        # Create keyword extractor
        self.keyword_extractor = Rake()
        self.keyword_vectors = []

        # Create average glove embedding vectors
        self.avg_glove_vectors = []
        if not glove_embeddings:
            return

        for doc in self.documents(main_col):
            avg_vector = np.mean([self.glove[str(word)] for word in doc.split() if str(word) in self.glove], axis=0)
            self.avg_glove_vectors.append(avg_vector)

            # Extract keywords with scores for every document
            self.keyword_extractor.extract_keywords_from_text(doc)
            phrases = self.keyword_extractor.get_ranked_phrases()
            words = [x for xs in phrases for x in xs.split()]

            # Get average glove embedding for every keyword
            keyword_vector = np.mean([self.glove[str(word)] for word in words if str(word) in self.glove], axis=0)
            self.keyword_vectors.append(keyword_vector)



    def documents(self, col:str = None) -> list:
        """
        Helper function to fetch an entire df column as a list, defaults to the main_col
        """
        col = self.main_col if col == None else col
        return self.df[col].to_list()

    def shape(self) -> str:
        return self.df.shape

    def save_df(self, path:str, sep = ',') -> str:
        self.df.to_csv(path, sep=sep, encoding='utf-8-sig')

    def n_rows(self) -> int:
        return len(self.df)

    def __repr__(self) -> str:
        return f"Dataframe: {self.shape()}, cols: {self.df.columns.names()}"


    # ==== Features ====

    def cosine_similarity_rank(self, query: str) -> list:
        """
        Returns the cosine similarity between a query string and every document in the dataframe

        Parameters
        ----------
        query: str, required
            the string to be queried
        """
        query_vector = self.vectorizer.transform([query]).toarray()
        return np.array([cosine_similarity([x],query_vector)[0][0] for x in self.fitted_vectorizer.toarray()])


    def overlapping_words_rank(self, query: str) -> np.ndarray:
        """
        Return a list of normalized scores for the amount of overlapping words a query string has in common with every document in the df.

        Parameters
        ----------
        query: str, required
            the string to be queried
        """
        overlapping_words = [0 for _ in range(len(self.documents()))]
        query_words = set(query.split())
        
        for i, doc in enumerate(self.documents()):
            n_overlap = len(query_words.intersection(set(doc.split())))
            overlapping_words[i] = n_overlap / min(len(query_words), len(doc))

        return np.array(overlapping_words)


    def nace_code_rank(self, query) -> np.ndarray:
        """
        Returns a list of nace code rankings between 0 and 1, that rank a query code against all other codes in the df which gives a higher score if the two code have more numbers in common.

        Parameters
        ----------
        query: str or int, required
            the string to be queried in the format '1234.0'
        """

        scores = [1 for _ in range(len(self.documents()))]
        
        # The NACE codes are in the format '1234.0', we dont want to include the .0
        query = str(query).split('.')[0]

        for i, code in enumerate(self.documents('NACE')):
            # Documents with nan NACE-codes will have a score of 1 normalized
            if np.isnan(code):
                continue
            
            # Score based on the number of characters they share
            for pos, (char1, char2) in enumerate(zip_longest(query, str(code).split('.')[0])):
                if char1 == char2:
                    scores[i] += (pos+1) + 1
                else:
                    break

        # Normalize the score to be between 0 and 1, and return
        max_score = max(scores)
        return np.array([x/max_score for x in scores])


    def keyword_rank(self, query: str) -> np.array:
        self.keyword_extractor.extract_keywords_from_text(query)
        phrases = self.keyword_extractor.get_ranked_phrases()
        query_words = [x for xs in phrases for x in xs.split()]

        # Get average glove embedding for every keyword
        query_vector = np.mean([self.glove[str(word)] for word in query_words if word in self.glove], axis=0)
        return np.array([cosine_similarity([query_vector],[i])[0][0] for i in self.keyword_vectors])


    def glove_rank(self, query: str) -> np.ndarray:
        """
        Gives the cosine similarity between the average glove embedding of a query string and every average glove embedding for df, which has been calculated beforehand

        Parameters
        ----------
        query: str, required
            the string to be queried
        """
        query_vector = np.mean([self.glove[word] for word in query.split() if word in self.glove], axis=0)
        return np.array([cosine_similarity([query_vector],[i])[0][0] for i in self.avg_glove_vectors])


    def word_feature(self, query:str) -> float:
        return len(query.split()) / self.avg_word_count

    def char_feature(self, query:str) -> float:
        return len(query) / self.avg_char_count

    def word_density_feature(self, query:str) -> float:
        return self.char_feature(query) / self.word_feature(query)


    def feature_vector(self, id: str) -> np.ndarray:
        """
        Given a company id, return the feature vector of that company using the cosine, overlap, glove and nace code features

        Parameters
        ----------
        id: str, required
            a company id string
        """
        # Safeguard against no query results or more than one query result (shouldnt happen tho)
        query = self.df[self.df.id == id]
        if len(query) == 0:
            print(f"No companies with id: {id} were found.")
            return

        query = query.iloc[0]
        text = query[self.main_col]

        cosine_rank = self.cosine_similarity_rank(text)
        overlap_rank = self.overlapping_words_rank(text)
        glove_rank = self.glove_rank(text)
        nace_rank = self.nace_code_rank(query['NACE'])
        keyword_rank = self.keyword_rank(text)
        
        return np.array([cosine_rank, overlap_rank, glove_rank, nace_rank, keyword_rank],dtype=object)


    def statistics_vector(self, id: str) -> np.ndarray:
                # Safeguard against no query results or more than one query result (shouldnt happen tho)
        query = self.df[self.df.id == id]
        if len(query) == 0:
            print(f"No companies with id: {id} were found.")
            return

        query = query.iloc[0]
        text = query[self.main_col]

        word_count = self.word_feature(text)
        char_count = self.char_feature(text)
        density_count = self.word_density_feature(text)
        return np.array([word_count, char_count, density_count])

    def get_tfidf_vectors(self) -> np.ndarray:
        return self.fitted_vectorizer.toarray()


if __name__ == "__main__":
    # Demo

    # Read in preprocessed data
    df = read_csv('data/processed_data/cleaned.csv')

    # Can be reused for multiple dataframes
    vectorizer = TfidfVectorizer(max_df=0.7,max_features=100)
    glove_df = read_csv('util/glove/glove.42B.300d.txt', sep=" ", index_col=0, quoting=3, header=None)
    glove_embeddings = {key: val.values for key, val in tqdm(glove_df.T.items())}
    # Delete glove_df to free up memory idk
    del glove_df

    dff = DataFrameFeatures(df=df, vectorizer=vectorizer, glove_embeddings=glove_embeddings)

    rank = dff.feature_vector(dff.df.id.iloc[0])
    print(rank)

