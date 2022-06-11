import numpy as np
from util.Util import read_csv
from tqdm import tqdm
import pandas as pd
from df_features import DataFrameFeatures
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class DataFrameClassifier:

    def __init__(self, dff: DataFrameFeatures, glove_embeddings: dict, file_name: str = None, verbose:bool = True) -> None:
        """

        """
        self.file_name = file_name

        subset_df = dff.df
        if file_name:
            input = self.read_train_file(file_name, verbose)
            subset_df = self.create_subset_df(dff.df, input)

        self.dff = DataFrameFeatures(df = subset_df, vectorizer=dff.vectorizer, glove_embeddings=glove_embeddings, main_col = dff.main_col, verbose=verbose)


    def create_subset_df(self, full_df: pd.DataFrame, input: pd.DataFrame) -> pd.DataFrame:
        """
        Creates a subset of the entire df that only includes the rows that are in the input file.
        Also adds the columns 'rating' and 'initial', which contain 0/1 and True/False values

        Parameters
        ----------
        full_df: dataframe, required
            the entire dataframe where the subset will be grabbed from

        input: dataframe, required
            the training file loaded as a csv which includes ids, ratings and initials
        """
        # Get the list of ids from the input df
        input.rename(columns={'Firmnav ID' : 'id'}, inplace=True)
        ids = list(set(input['id'].tolist()))
        
        # Get a subset of the entire df based on the ids
        df = full_df[full_df.id.isin(ids)]
        df = df.drop_duplicates(subset=['id'],keep='first')

        # Merge the subset df and the input df to get a initial and rating col
        df = df.merge(input, on='id', how='left')

        # Clean up and renaming
        df['AI search'] = list(df['AI search'] == "Initial")
        df.rename(columns={'AI search' : 'Initial'}, inplace=True)

        return df


    def read_train_file(self, file_name: str, verbose:bool) -> pd.DataFrame:
        """
        Read a training file with a given filename.
        Also removes all entires without ratings
        """
        df = read_csv(f'data/train/{file_name}.csv', sep=';', verbose=verbose)
        # Remove all entries that dont have a rating and drop any duplicate ids
        df = df[~df.Rating.isnull()]
        df = df.drop_duplicates(subset=['Firmnav ID'],keep='first')
        return df

    def get_feature_vectors(self, split: bool = False) -> tuple:
        """
        Create feature vectors and labels (X,y) for all documents in the dataframe.

        Parameters
        ----------
        split: bool, optional
            A bool indicating whether the function should split the data into train and test sets,
            or return it as one set.
            If true the dataframe is split, where the initials are used as the test-set
            and the rest are returned as the training-set.
        """
        feature_vectors = np.array([self.dff.feature_vector(doc_id) for doc_id in self.dff.documents('id')])
        X_means = np.mean(feature_vectors,axis=0)
        X_vars = np.var(feature_vectors, axis=0)
        X_q25 = np.quantile(feature_vectors,q=0.25,axis=0)
        X_medians = np.median(feature_vectors, axis=0)
        X_q75 = np.quantile(feature_vectors,q=0.75,axis=0)
        # idf = self.dff.vectorizer.idf_.reshape(1,-1) / np.mean(self.dff.vectorizer.idf_)
        X_1 = np.vstack((X_means, X_vars, X_q25, X_medians, X_q75)).T

        # X_1 = np.array([np.mean(self.dff.feature_vector(doc_id),axis=1) for doc_id in self.dff.documents('id')])
        X_2 = np.array([self.dff.statistics_vector(doc_id) for doc_id in self.dff.documents('id')])
        X = np.hstack((X_1, X_2))
        y = np.array(self.dff.documents('Rating'))
        

        if not split:
            return (X, y)

        filter = self.dff.df['Initial'].to_numpy()
        X_train, y_train = X[~filter], y[~filter]
        X_test, y_test = X[filter], y[filter]
        
        return X_train, y_train, X_test, y_test


    def get_tfidf_vectors(self, split:bool = False) -> tuple:
        X = self.dff.get_tfidf_vectors()
        y = np.array(self.dff.documents('Rating'))

        if not split:
            return (X, y)

        filter = self.dff.df['Initial'].to_numpy()
        X_train, y_train = X[~filter], y[~filter]
        X_test, y_test = X[filter], y[filter]
        
        return X_train, y_train, X_test, y_test



# Example usage
if __name__ == "__main__":
    # Read in preprocessed data
    df = read_csv('data/processed_data/cleaned.csv')

    # Can be reused for multiple dataframes
    vectorizer = TfidfVectorizer(max_df=0.7,max_features=100)
    glove_df = read_csv('util/glove/glove.42B.300d.txt', sep=" ", index_col=0, quoting=3, header=None)
    glove_embeddings = {key: val.values for key, val in tqdm(glove_df.T.items())}
    
    del glove_df # Delete glove_df to free up memory

    dff = DataFrameFeatures(df=df, vectorizer=vectorizer)
    dfc = DataFrameClassifier(dff=dff, glove_embeddings=glove_embeddings, file_name="Consulting")

    X_train, y_train, X_test, y_test  = dfc.get_feature_vectors(split=True)

    print(X_train,y_train)


