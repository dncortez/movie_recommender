import numpy as np
import pandas as pd
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MultiLabelBinarizer
from nltk.tokenize import word_tokenize

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def cosine_similarity(x, y, smooth = True):
    dot = np.sum(x*y, axis = -1)
    norm = (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)) + 1e-320 if smooth else 0
    return dot/norm
    
class OtherTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 10):
        self.threshold = threshold
        self.dict = {}
    def fit(self, X, y=None):
        self.dict = {}
        for col in X:
            vc = X.value_counts() < self.threshold
            self.dict[col] = {cat[0]: "Other" if cond else cat[0]
                              for cat, cond in zip(vc.index, vc)}
        return self
    def transform(self, X):
        X = X.copy()
        for col in self.dict:
            X[col] = X[col].apply(lambda x: self.dict[col][x])
        return X
    def set_output(self, *args, **kwargs):
        pass
        
class SplitTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sep = ","):
        self.sep = sep
        self.mlb = {}
    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        X = X.applymap(lambda x: x.split(self.sep) if type(x) is str else [])
        for col in X:
            self.mlb[col] = MultiLabelBinarizer().fit(X[col])
        return self
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        X = X.applymap(lambda x: x.split(self.sep) if type(x) is str else [])
        return np.hstack([self.mlb[col].transform(X[col]) for col in X])
    def set_output(self, *args, **kwargs):
        pass

class TextTransformer():
    def __init__(self, corpus, preprocessor, seq_length = 1000, vocab = None, vocab_to_int = None):

        self.preprocess = preprocessor
        
        if vocab is None:
            corpus= [
                txt.lower()
                for text in corpus
                for txt in self.preprocess(text)
            ]

            self.vocab = list(set(corpus))
        else:
            self.vocab = vocab
        self.vocab_to_int = {word:idx for idx, word in enumerate(self.vocab, 1
                                                                )} if vocab_to_int is None else vocab_to_int

        self.seq_length = seq_length

    def __call__(self, words):
        ints = np.zeros(self.seq_length, dtype=int)
        
        words = self.preprocess(words)[:self.seq_length]
        
        ints[:min(len(words), self.seq_length)] = np.array([
            self.vocab_to_int[word.lower()] if word.lower() in self.vocab_to_int else 0
            for word in words])

        return (ints, len(words))
        
class MovieDataset:
    def __init__(self, features, text, movie_ids, barcode, transform= None, text_transformed = False):
        self.transform = transform
        
        self.movie_ids = movie_ids
        self.movie_id_to_idx = {int(movie_id):idx for idx, movie_id in enumerate(movie_ids)}
        
        self.features = np.array(features)
        
        if text_transformed:
            self.text, self.lengths = text
        else:
            self.text, self.lengths = zip(*[self.transform(txt) for txt in text])
            self.text = np.array(self.text)
            self.lengths = np.array(self.lengths)
        
        self.barcode = np.array(barcode)
    def __len__(self):
        return self.features.shape[0]
    def get_data(self, movie_id):
        idx = self.movie_id_to_idx[movie_id]
        
        return self[idx]
    def __getitem__(self, idx):
        text     = self.text[idx]
        lengths   = self.lengths[idx]
        features = self.features[idx]
        
        text     = torch.tensor(text)
        lengths  = torch.tensor(lengths)
        features = torch.tensor(features, dtype = torch.float32)
        
        return (text, features, lengths)
    
    def cat(self, other):
        new = movie_dataset_2(self.features, (self.text, self.lengths), self.movie_ids, self.barcode,
                             transform = self.transform, text_transformed = True)
        new.text       = np.concatenate((new.text,       other.text))
        new.features   = np.concatenate((new.features,   other.features))
        new.lengths    = np.concatenate((new.lengths,    other.lengths))
        new.barcode    = np.concatenate((new.barcode,    other.barcode))
        new.movie_ids  = np.concatenate((new.movie_ids,  other.movie_ids))
        for movie_id in other.movie_id_to_idx:
            new.movie_id_to_idx[movie_id] = other.movie_id_to_idx[movie_id] + len(self)
        return new
        
def interaction_matrix(interactions, movies):
    out = pd.DataFrame(0, columns = interactions.userId.unique(), index = movies.index)
    for row in interactions.iloc[:, :2].to_numpy():
        out.loc[row[1], row[0]]+=1
    return out
    

class MovieNet2(nn.Module):
    def __init__(self, embedding_dim, vocab_size, lstm_dim, lstm_layers, n_features, 
                 hidden_dim, output_dim, drop_prob=0.2, bidirectional = False):
        
        super(MovieNet2, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, lstm_dim, lstm_layers,
                            dropout=drop_prob, batch_first=True, bidirectional = bidirectional)

        self.dropout = nn.Dropout(drop_prob)
        
        # Capas lineales y salida
        self.fc1 = nn.Linear(lstm_dim*(1+bidirectional) + n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.fun = nn.ReLU()

    def forward(self, text, features, lengths):
        
        x = self.embedding(text)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)
        x, _ = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)   
        
        #Tomamos solo el último valor de salida del LSTM
        x = x[:,-1,:]
        
        # Concatenamos con features
        x = torch.hstack((x, features))
        x = self.dropout(x)
        
        # Capas finales y salida
        x = self.fc1(x)
        x = self.fun(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fun(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
        
class MovieNet3( MovieNet2):
    def __init__(self, embedding_dim, vocab_size, lstm_dim, lstm_layers, n_features, 
                 hidden_dim, output_dim, drop_prob=0.2, bidirectional = False):
        
        super(MovieNet3, self).__init__(embedding_dim, vocab_size, lstm_dim, lstm_layers, n_features, 
                 hidden_dim, output_dim, drop_prob=drop_prob, bidirectional = bidirectional)

    def forward(self, text, features, lengths):
        x = self.embedding(text)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first = True, enforce_sorted = False)
        x, _ = self.lstm(x)
        x, lengths = nn.utils.rnn.pad_packed_sequence(x, batch_first = True)   
        
        #Tomamos solo el último valor de salida del LSTM
        x = x[:,-1,:]
        
        # Concatenamos con features
        x = torch.hstack((x, features))
        x = self.dropout(x)
        
        # Capas finales y salida
        x = self.fc1(x)
        x = self.fun(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x
    
class UserNet(nn.Module):
    def __init__(self, input_dim, user_embedding_dim, movie_embedding_dim,
                 hidden_dim_1, hidden_dim_2, drop_prob=0.2):
        
        super(UserNet, self).__init__()
        
        self.fc_in_1    = nn.Linear(input_dim,    hidden_dim_1)
        self.fc_in_2    = nn.Linear(hidden_dim_1, user_embedding_dim)
        self.fc_out_1   = nn.Linear(user_embedding_dim + movie_embedding_dim, hidden_dim_2)
        self.fc_out_2   = nn.Linear(hidden_dim_2, 1)

        self.dropout = nn.Dropout(drop_prob)
        
        self.fun = nn.ReLU()
        
    def embed(self, x):
        #x = self.dropout(x)
        x = self.fc_in_1(x)
        x = self.fun(x)
        x = self.dropout(x)
        return self.fc_in_2(x)
    
    def likeliness(self, user_input, movie_embeddings):
        user_input = torch.tile(user_input.unsqueeze(1), (movie_embeddings.shape[0], 1))
        movie_embeddings = torch.tile(movie_embeddings.unsqueeze(0), (user_input.shape[0], 1, 1))
        x = torch.cat((movie_embeddings, user_input), 2)
        
        x = self.dropout(x)
        x = self.fc_out_1(x)
        x = self.fun(x)
        x = self.dropout(x)
        x = self.fc_out_2(x).squeeze()
        return torch.sigmoid(x)

    def forward(self, user_input, movie_embeddings):
        user_input = self.embed(user_input)
        return self.likeliness(user_input, movie_embeddings)
        
    
class Recommender:
    def __init__(self, movienet, usernet, interactions, dataset, user_barcode_index, batch_size = 20,
                movie_embeddings = None):
        self.movienet = movienet
        self.usernet = usernet
        self.interactions = interactions
        self.dataset = dataset
        self.user_barcode_index = user_barcode_index
        self.movie_ids = dataset.movie_ids
        self.movie_embeddings = movie_embeddings
        if movie_embeddings is None:
            self.movie_embeddings = self.get_movie_embeddings(batch_size = batch_size)
    def get_movie_embeddings(self, batch_size = 20):
        self.movienet.to(DEVICE)
        self.movienet.eval()
        movie_ids = self.movie_ids
        embeds = []
        bar = tqdm(range(0, len(movie_ids), batch_size))
        bar.set_description_str("Calculating movie embeddings")
        for i in range(0, len(movie_ids), batch_size):
            text, features, lengths = self.dataset[i:i+batch_size]
            embed = self.movienet(text.to(DEVICE), features.to(DEVICE), lengths)
            embeds += [embed.cpu().detach().numpy()]
            bar.update()
        return np.vstack(embeds)
    
    def get_new_embedding(self, movie_ids, mean = True):
        indexs = [self.dataset.movie_id_to_idx[movie_id] for movie_id in movie_ids]
        embeddings = self.movie_embeddings[indexs]
        return embeddings.mean(axis =0) if mean else embeddings
    
    def recommend(self, user_id = None, movie_ids = None, from_embedding=None, n = "auto", 
                  sigma = 0.2, mean = False, include_input = False):
        assert not (user_id is None and movie_ids is None and from_embedding is None), "Either specify a user_id or a list of movie_ids"
        if from_embedding is None:
            if user_id is not None:
                movie_ids = self.interactions[self.interactions.userId == user_id].movieId
            user_embeddings = self.get_new_embedding(movie_ids, mean=mean)
            n = len(movie_ids) if n == "auto" else n
        else:
            user_embeddings = from_embedding

        all_movie_embeddings = np.expand_dims(self.movie_embeddings, 1)
        user_embeddings      = np.expand_dims(user_embeddings, 0)
        similarities = cosine_similarity(user_embeddings, all_movie_embeddings)
        similarities = np.exp( -(1 - similarities)**2 / (2*sigma**2))
        similarities = similarities.sum(axis =-1)

        order = np.argsort(-similarities)
        all_movie_ids = self.movie_ids.copy()[order]
        if (not include_input) and from_embedding is None:
            all_movie_ids = all_movie_ids[~np.isin(all_movie_ids, movie_ids)]
        return np.array(all_movie_ids[:n])
    
    def recommend2(self, user_id = None, movie_ids = None, from_embedding=None, n = "auto", 
                  include_input = False):
        assert not (user_id is None and movie_ids is None and from_embedding is None), "Either specify a user_id or a list of movie_ids"
        if from_embedding is None:
            if user_id is not None:
                movie_ids = self.interactions[self.interactions.userId == user_id].movieId
            input_vector = pd.Series(0, index = self.user_barcode_index)
            input_vector[movie_ids] = 1
            input_vector = torch.tensor(input_vector.to_numpy().copy(), dtype = torch.float32)
            self.usernet.eval()
            user_embedding = self.usernet.embed(input_vector.unsqueeze(0).to(DEVICE))
            
            n = len(movie_ids) if n == "auto" else n
        else:
            user_embedding = torch.tensor(from_embedding, dtype = torch.float32).to(DEVICE)
        
        movie_embeddings_tensor = torch.tensor(self.movie_embeddings, dtype = torch.float32).to(DEVICE)
        outputs = self.usernet.likeliness(user_embedding, movie_embeddings_tensor)
        outputs = outputs.detach().cpu().numpy()

        order = np.argsort(-outputs)
        all_movie_ids = self.movie_ids.copy()[order]
        if (not include_input) and from_embedding is None:
            all_movie_ids = all_movie_ids[~np.isin(all_movie_ids, movie_ids)]
        return np.array(all_movie_ids[:n])

    def accuracy(self, user_id = None, movie_ids = None, sigma = 0.2, method = 1,
                 mean = False, include_input= False, input_ratio = 0.5):
        assert not (user_id is None and movie_ids is None), "Either specify a user_id or a list of movie_ids"
        if user_id is not None:
            movie_ids = self.interactions[self.interactions.userId == user_id].movieId
        movie_ids = np.array(movie_ids)
        movie_ids_in = np.random.choice(movie_ids, int(len(movie_ids)*input_ratio), replace= False)
        if include_input:
            movie_ids_target = movie_ids.copy()
        else: 
            movie_ids_target = movie_ids[~np.isin(movie_ids, movie_ids_in)]
        
        if method == 1:
            recommendations = self.recommend(movie_ids = movie_ids_in, n=len(movie_ids_target),
                                             sigma=sigma, mean = mean, include_input = include_input)
        else:
            recommendations = self.recommend2(movie_ids = movie_ids_in, n=len(movie_ids_target),
                                             include_input = include_input)
            
        return len(set(recommendations).intersection(set(movie_ids_target))) / len(set(recommendations))