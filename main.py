from fastapi import FastAPI

from utils import *

app = FastAPI()
    
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

movies = pd.read_csv("movies.csv", index_col = 0)
movies = movies.set_index("movieId")

interactions = pd.read_csv("interactions.csv", index_col=0)
        
ct = ColumnTransformer([
    ("Normalize", StandardScaler(), ["Release Year"]),
    ("Multilabel", SplitTransformer("|"), ["genres"]),
    ("OneHotOther", Pipeline([
        ("Other", OtherTransformer(threshold = 5)),
        ("OneHot", OneHotEncoder(sparse_output=False)),
    ]), ["Origin/Ethnicity"])
], remainder = "drop")

        
#tt = TextTransformer(movies_plot_train, word_tokenize)
tt = TextTransformer([], word_tokenize, vocab = list(np.load("weights/VocabToInt.npy", allow_pickle = True).item().keys()),
                     vocab_to_int = np.load("weights/VocabToInt.npy", allow_pickle = True).item())

movies_data = ct.fit_transform(movies)

interactions_train = interactions[interactions.userId<=530]
interactions_test  = interactions[interactions.userId>530]

#barcoding = interaction_matrix(interactions_train, movies)
barcoding = pd.read_csv("weights/Barcoding.tsv", sep = "\t", index_col = 0)

#full_dataset = MovieDataset(movies_data, movies.Plot, movies.index, barcoding, transform = tt)
dataset_text    = np.load("weights/MovieDataset.text.npy")
dataset_lengths = np.load("weights/MovieDataset.lengths.npy")
full_dataset = MovieDataset(movies_data, (dataset_text, dataset_lengths), 
                            movies.index, barcoding, transform = tt, text_transformed = True)


movienet1 = MovieNet3(256, len(tt.vocab)+1, 128, 2, 32, 128, 16, drop_prob=0.4)
movienet1.to(DEVICE)
movienet1.load_state_dict(torch.load("weights/MovieNet1.pt", map_location = DEVICE))
movienet2 = MovieNet3(256, len(tt.vocab)+1, 128, 2, 32, 128, 32, drop_prob=0.2)
movienet2.to(DEVICE)
movienet2.load_state_dict(torch.load("weights/MovieNet2.pt", map_location = DEVICE))

usernet1 = UserNet(barcoding.shape[0], 64, 16, 256, 64, drop_prob = 0.0)
usernet1.to(DEVICE)
usernet1.load_state_dict(torch.load("weights/UserNet1.pt", map_location = DEVICE))
usernet2 = UserNet(barcoding.shape[0], 64, 32, 256, 128, drop_prob = 0.0)
usernet2.to(DEVICE)
usernet2.load_state_dict(torch.load("weights/UserNet2.pt", map_location = DEVICE))

recommender1 = Recommender(movienet1, usernet1, interactions, full_dataset, barcoding.index, batch_size = 50,
                           movie_embeddings = np.load("weights/MovieEmbeddings1.npy"))
recommender2 = Recommender(movienet2, usernet2, interactions, full_dataset, barcoding.index, batch_size = 50,
                           movie_embeddings = np.load("weights/MovieEmbeddings2.npy"))
                      
                      


@app.get("/get_new_movie_recommendation")
def get_new_movie_recommendation(model:int = 2, year: int = 1992, origin:str = "American", genres: str = "(no genres listed)", plot: str = ".", n:int = 8):
    recommender_system = recommender1 if model==1 else recommender2

    genres = "|".join(genres.split(","))

    df = pd.DataFrame({"Release Year": [year], "Origin/Ethnicity": [origin], "genres": [genres]})
    features = ct.transform(df)
    text, length = tt(plot)
    net_in = (torch.tensor(text, dtype = torch.int32).cuda().unsqueeze(0),
              torch.tensor(features, dtype = torch.float32).cuda(),
              torch.tensor(length).unsqueeze(0))
    recommender_system.movienet.eval()
    embedding = recommender_system.movienet(*net_in).cpu().detach().numpy()
    movie_ids = recommender_system.recommend(from_embedding = embedding, n = n, sigma = 0.1)

    return list(movies.loc[movie_ids].Title)

@app.get("/get_new_user_recommendation")
def get_new_user_recommendation(movieIds:str, model:int = 2, n:int = 8):
    movie_ids = [int(number) for number in movieIds.split(",")]
    recommender_system = recommender1 if model==1 else recommender2
    
    recommendations = recommender_system.recommend2(movie_ids = movie_ids, n=n)
    
    return list(movies.loc[recommendations].Title)
    
@app.get("/get_current_user_recommendation")
def get_current_user_recommendation(userId:int, model:int = 2, n:int = 8):
    recommender_system = recommender1 if model==1 else recommender2
    
    recommendations = recommender_system.recommend2(user_id = userId, n=n)
    
    return list(movies.loc[recommendations].Title)
