import numpy as np
import pandas as pd
import fasttext
# from sklearn import linear_model
from langdetect import detect
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import random
from annoy import AnnoyIndex
from numba import jit
from sklearn.cluster import KMeans
# from sklearn.linear_model import LassoCV
# from sklearn.feature_selection import SelectFromModel
# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.linear_model import LogisticRegression
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import GridSearchCV
# from sklearn.linear_model import ElasticNet
# from sklearn.neighbors import NearestNeighbors
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error
# from sklearn import preprocessing
from math import sqrt
import psycopg2
import faiss

## Helper methods
@jit
def create_index(length):
    temp = []
    for i in range(length):
        temp.append(i)
    temp = np.array(temp)
    return temp

@jit
def adding_to(t, df):
    for i in range(len(df)):
        temp = df[i]
        t.add_item(i, temp)

def turn_link(string):
    return 'https://open.spotify.com/track/' + str(string)

## Load translator
def load_trans():
    ## Fasttext model
    model_loc = 'lid.176.bin'
    model = fasttext.load_model(model_loc)
    return model


## SQL thingd

def connect_to_df(db_name):
    """ Connect to the PostgreSQL database server """
    conn = None
    try:
        # read connection parameters

        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host="localhost",
            database=db_name,
            user="postgres",
            password="Noobs4321")
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    print("Connection Success")
    return conn

def postgres_to_df(conn, column_names, table_name):
    statement = "SELECT * FROM " + table_name
    tupples = query(statement, conn)
    tracks = pd.DataFrame(tupples, columns= column_names)
    print("SQL to DF success")
    return tracks

def query(statement, conn):
    cur = conn.cursor()
    try:
        cur.execute(statement)
    except (Exception, psycopg2.DatabaseError) as error:
        print("Error: %s" % error)
        cur.close()
        return 1

    tupples = cur.fetchall()
    print("Query Success")
    cur.close()
    return tupples

## Clustered + KNN
def model2(query):
    ## Load data
    connect = connect_to_df("MusicRecommender")
    tracks = postgres_to_df(connect, important, "musiclist")
    tracks = tracks[important].dropna()

    ## Only English songs
    tracks['name'] = tracks.name.str.encode('utf-8')
    tracks['name'] = tracks['name'].astype('|S')
    lang = []
    lang_model = load_trans()
    for i in tracks['name']:
        try:
            lang.append(lang_model.predict(i.decode("utf-8"))[0])
        except Exception:
            lang.append('Error')
    print("done converting")
    tracks['lang'] = lang
    # tracks.to_csv("AllLang.csv", index=False)
    tracks['lang'] = tracks['lang'].astype('str')
    temp = tracks.loc[tracks['id'] == query]  # adding a row
    tracks = tracks[tracks['lang'].str.contains('en')]
    tracks = tracks.append(temp, ignore_index=True)
    tracks.to_csv("English.csv", index=False)
    tracks = tracks.dropna(axis=0)

    ## Scaling data
    scaler = StandardScaler().fit(tracks[features_before])
    scaled_data = scaler.transform(tracks[features_before])
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data.columns = tracks[features_before].columns
    scaled_data['mode'] = tracks['mode']
    scaled_data['time_signature'] = tracks['time_signature']
    scaled_data['explicit'] = tracks['explicit']
    scaled_data['id'] = tracks['id']
    scaled_data['name'] = tracks['name']

    model2 = scaled_data.copy()
    model2['index'] = create_index(len(model2))
    model2.drop(['energy'], axis=1)
    model2 = model2.dropna(axis=0)

    ## Clustering
    km = KMeans(n_clusters=25)
    predicted_genres = km.fit_predict(model2[features_model2])
    model2['cluster_assignment'] = predicted_genres
    model2.to_csv("Clustered.csv", index=False)
    print("Done training")

    ## Spotify KNN
    print('starting query')

    ## Finding song
    selected_track = model2.loc[model2['id'] == query]
    selected_track = selected_track["cluster_assignment"]
    list_tracks = model2[model2['cluster_assignment'] == selected_track.values[0]]
    list_tracks = list_tracks.reset_index(drop=True)
    tracks_query = list_tracks[features_model2]
    list_tracks.to_csv('ClusteredSongs.csv', index=False)

    ## Training Spotify ANNOY
    f = len(features_model2)
    t = AnnoyIndex(f, 'angular')
    adding_to(t, tracks_query.to_numpy())
    print('done adding')
    t.build(20)  # 20 trees
    t.save('test.ann')

    u = AnnoyIndex(f, 'angular')
    u.load('test.ann')

    ## Search for closet songs and save to csv
    search_num = list_tracks[list_tracks['id'].str.contains(query)].index[0]
    print(search_num)
    similar_songs = u.get_nns_by_item(search_num, 10)
    similar_songs = list_tracks.iloc[similar_songs]
    similar_songs['id'] = similar_songs['id'].apply(turn_link)
    temp = similar_songs['id'].to_numpy()
    print(temp)
    np.savetxt('song.txt', temp, fmt='%s', delimiter=',')


important = ['id', 'name', 'popularity', 'duration_ms', 'explicit', 'artists', 'id_artists', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'popularity_bin']
features_knn = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'popularity_bin']
features = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
features_model2 = ['duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'popularity']
features_before = ['duration_ms', 'danceability', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'popularity']
target = ['popularity_bin']

model2('4saklk6nie3yiGePpBwUoc')
# connect = connect("MusicRecommender")
# tracks = postgres_to_df(connect, important, "musiclist")
# tracks = tracks[important].dropna()


## Get language

# tracks['name'] = tracks.name.str.encode('utf-8')
# tracks['name'] = tracks['name'].astype('|S')
# lang = []
# for i in tracks['name']:
#     try:
#         lang.append(model.predict(i.decode("utf-8"))[0])
#     except Exception:
#         lang.append('Error')
# print("done converting")
# tracks['lang'] = lang
# #tracks.to_csv("AllLang.csv", index=False)
# tracks['lang'] = tracks['lang'].astype('str')
# temp = tracks.loc[tracks['id'] == "7dt6x5M1jzdTEt8oCbisTK"] # adding a row
# tracks = tracks[tracks['lang'].str.contains('en')]
# tracks = tracks.append(temp, ignore_index = True)
# # tracks.loc[-1] = temp
# # tracks.index = tracks.index + 1  # shifting index
# # tracks = tracks.sort_index()  # sorting by index
#
# tracks.to_csv("English.csv", index=False)
# tracks = tracks.dropna(axis=0)

## Scaling data
# scaler = StandardScaler().fit(tracks[features_before])
# scaled_data = scaler.transform(tracks[features_before])
# scaled_data = pd.DataFrame(scaled_data)
# scaled_data.columns = tracks[features_before].columns
# scaled_data['mode'] = tracks['mode']
# scaled_data['time_signature'] = tracks['time_signature']
# scaled_data['explicit'] = tracks['explicit']
# scaled_data['id'] = tracks['id']
# scaled_data['name'] = tracks['name']

# model1 = scaled_data.copy()
# model1['index'] = create_index(len(model1))
# model2 = scaled_data.copy()
# model2['index'] = create_index(len(model2))
# model2.drop(['energy'], axis=1)
# model2 = model2.dropna(axis=0)

## Generate random songs for model2
# np.random.seed(544)
# index = np.arange(model2.shape[0])
# random_list = np.random.choice(index, size=20, replace=False)
# random_songs = model2.iloc[random_list]
# random_songs_save = []
# song_ids = []
# song_index =[]
# for id in random_songs['id']:
#     random_songs_save.append('https://open.spotify.com/track/{}'.format(id))
#     song_ids.append(id)
# for index in random_songs['index']:
#     song_index.append(index)
#
#
# random_songs.to_csv("ListSongs.csv", index=False)
# a_file = open("test.txt", "w+")
# content = str(random_songs_save)
# a_file.write(content)
# a_file.close()
# selected_ids = ['3SSkMFL1miBxUfPGaaL5W4', '161DnLWsx1i3u1JT05lzqU', '6GiFcHzZtT7NB7ZPk4ZmEd', '7BqBn9nzAq8spo5e7cZ0dJ', '0ZkZmomLyBHJKRGA0Uv9Su' ,'0RUGuh2uSNFJpGMSsD1F5C', '4d2nmevJcws03HNpYiP4FZ', '2tJulUYLDKOg9XrtVkMgcJ', '0bHpsorTpt9wX6Dkr2LEfg', '5KawlOMHjWeUjQtnuRs22c', '78gDp7tT1QIUwVJwNYXxOR', '2iuZJX9X9P0GKaE93xcPjk', '2bL2gyO6kBdLkNSkxXNh6x', '6ECp64rv50XVz93WvxXMGF', '72Lk8C1Eux8P0hahS3uesA', '2W4yABdLc4Jnds67Poi5Nl', '4a6q8CR2hzLk2plDkSxkfD', '696DnlkuDOXcMAnKlTgXXK', '5dn6QANKbf76pANGjMBida', '7ivYWXqrPLs66YwakDuSim']
# selected_songs = model2[model2['id'].isin(selected_ids)]
# selected_songs.to_csv("Selected.csv", index=False)
#
# training_df = [2.3, 0, 3.2, 0, 0, 6, 4, 0, 8, 4, 6, 0, 8.9, 3, 5, 6.4, 3.4, 7, 9, 10]
# training_df2 = [0, 10, 0, 10, 0, 10, 0, 10, 2.3, 10, 3.4, 10, 10, 0, 10, 10, 10, 10, 10, 10]

## Training LR
# simple_LR = LinearRegression().fit(selected_songs[features_model2], training_df2)
#
# weights = zip(selected_songs[features_model2].columns.values, simple_LR.coef_)
# for weight in weights:
#     print(weight)

## drop new NA
# model2 = model2.dropna(axis=0)
# predictionsLR = simple_LR.predict(model2[features_model2])
#
# resultsLR = model2.copy()
# resultsLR['predicted_ratings'] = predictionsLR
#
# results_sortedLR = resultsLR.sort_values(by=['predicted_ratings'])
# top10LR = results_sortedLR[results_sortedLR.shape[0] - 10:]
# top10LR.to_csv("PredictedLR.csv", index=False)
# print("predicted 10 LR")
#
# ## Training RF
# simple_RF = RandomForestRegressor(random_state=15).fit(selected_songs[features_model2], training_df2)
# predictionsRF = simple_RF.predict(model2[features_model2])
# resultsRF = model2.copy()
# resultsRF['predicted_ratings'] = predictionsRF
#
# results_sortedRF = resultsRF.sort_values(by=['predicted_ratings'])
# top10RF = results_sortedRF[results_sortedRF.shape[0] - 10:]
# top10RF.to_csv("PredictedRF.csv", index=False)
# print("predicted 10 RF")



# print("Testing")
# search = query.loc[query["id"] == "5EcpuCZ75oXmU5j0Hqoyys"]
# search = search["id"]
# # temp = query[query.equals(search)]
# print(search.values[0])
# print(temp)

# conditions = [
#     tracks['popularity'] <= 25,
#     (tracks['popularity'] > 25) & (tracks['popularity'] <= 50),
#     (tracks['popularity'] > 50) & (tracks['popularity'] <= 75),
#     (tracks['popularity'] > 75) & (tracks['popularity'] <= 100)
# ]
#
# values = [
#     1,
#     2,
#     3,
#     4
# ]
# tracks['popularity_bin'] = np.select(conditions, values, default=0)

## Random selection
# indexs = random.sample(range(len(model1)), 12000)
# val_indexs = indexs[10000:12000]
# indexs = indexs[0:10000]
#
# training = model1.iloc[indexs]
# validation = model1.iloc[val_indexs]

## Normalizing data
# x = temp.values
# scaler = preprocessing.MinMaxScaler()
# x_scaled = scaler.fit_transform(x)
# df = pd.DataFrame(x_scaled)

## Training models for popularity
## AdaBoost
# data = []
# for i in range(1, 15):
#     regr = AdaBoostClassifier(n_estimators=i, random_state=0)
#     regr.fit(training[features], training[target])
#     y_pred_train = regr.predict(training[features])
#     y_pred_val = regr.predict(validation[features])
#     for i, s in enumerate(y_pred_train):
#         y_pred_train[i] = np.round(y_pred_train[i])
#     y_pred_train = y_pred_train.astype(np.int)
#     for i, s in enumerate(y_pred_val):
#         y_pred_val[i] = np.round(y_pred_val[i])
#     y_pred_val = y_pred_val.astype(np.int)
#     train_rmse = sqrt(mean_squared_error(training[target], y_pred_train))
#     validation_rmse = sqrt(mean_squared_error(validation[target], y_pred_val))
#     data.append({
#         'max_depth': i,
#         'model': regr,
#         'train_rmse': train_rmse,
#         'validation_rmse': validation_rmse
#     })
# ada_data = pd.DataFrame(data)
# print(ada_data)

## DecisionTree
# hyperparameters = {'min_samples_leaf': [1, 10, 50, 100, 200, 300], 'max_depth': [1, 2, 3, 4, 5, 10, 15, 20]}
# data = []
# for i in range(2, 10):
#     search = GridSearchCV(DecisionTreeClassifier(), hyperparameters, cv=i, return_train_score=True)
#     search = search.fit(training[features], training[target])
#     y_pred_train = search.predict(training[features])
#     y_pred_val = search.predict(validation[features])
#     for i, s in enumerate(y_pred_train):
#         y_pred_train[i] = np.round(y_pred_train[i])
#     y_pred_train = y_pred_train.astype(np.int)
#     for i, s in enumerate(y_pred_val):
#         y_pred_val[i] = np.round(y_pred_val[i])
#     y_pred_val = y_pred_val.astype(np.int)
#     train_rmse = sqrt(mean_squared_error(training[target], y_pred_train))
#     validation_rmse = sqrt(mean_squared_error(validation[target], y_pred_val))
#     data.append({
#         'cv': i,
#         'model': search,
#         'train_rmse': train_rmse,
#         'validation_rmse': validation_rmse
#     })
#
# decision_tree_data = pd.DataFrame(data)
# print(decision_tree_data)

## Logistic
# l2_penalties = [0.01, 1, 4, 10, 1e2, 1e3, 1e5]
# l2_penalty_names = [f'coefficients [L2={l2_penalty:.0e}]'
#                     for l2_penalty in l2_penalties]
# accuracy_data = []
# for x in range(len(l2_penalties)):
#     model = LogisticRegression(penalty='l2', random_state=1, C=1/l2_penalties[x], fit_intercept=False)
#     model.fit(training[features], training[target])
#     y_pred_train = model.predict(training[features])
#     y_pred_val = model.predict(validation[features])
#     for i, s in enumerate(y_pred_train):
#         y_pred_train[i] = np.round(y_pred_train[i])
#     y_pred_train = y_pred_train.astype(np.int)
#     for i, s in enumerate(y_pred_val):
#         y_pred_val[i] = np.round(y_pred_val[i])
#     y_pred_val = y_pred_val.astype(np.int)
#     model_train_accuracy = accuracy_score(training[target], y_pred_train)
#     model_validation_accuracy = accuracy_score(validation[target], y_pred_val)
#     accuracy_data.append({
#         'max_depth': x,
#         'model': model,
#         'train_rmse': model_train_accuracy,
#         'validation_rmse': model_validation_accuracy
#     })
# accuracy_data = pd.DataFrame(accuracy_data)
# print(accuracy_data)

## ElasticNet
# parametersGrid = {"max_iter": [1, 5, 10],
#                       "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                       "l1_ratio": np.arange(0.0, 1.0, 0.1)}
# elastic = ElasticNet()
# grid = GridSearchCV(elastic, parametersGrid, scoring='r2', cv=10)
# grid.fit(training[features], training[target])
# elasticv2_training = grid.predict(training[features])
# elasticv2_validation = grid.predict(validation[features])
# for i, s in enumerate(elasticv2_training):
#     elasticv2_training[i] = np.round(elasticv2_training[i])
# elasticv2_training = elasticv2_training.astype(np.int)
# for i, s in enumerate(elasticv2_validation):
#     elasticv2_validation[i] = np.round(elasticv2_validation[i])
# elasticv2_validation = elasticv2_validation.astype(np.int)
# print("Training: ", sqrt(mean_squared_error(training[target], elasticv2_training)), " | ", "Validation: ", sqrt(mean_squared_error(validation[target], elasticv2_validation)))

## Visualizing correlation

#corr = tracks.corr()
#plt.figure(figsize=(20,8))
#sns.heatmap(corr, vmax=1, vmin=-1, center=0,linewidth=.5,square=True, annot = True, annot_kws = {'size':8},fmt='.1f', cmap='coolwarm')
#plt.title('Correlation')
#plt.show()

## Normalizing columns
# tracks_scaled = pd.DataFrame()
# scaler = MinMaxScaler()
# for col in query[features].columns:      # excluding year col i.e, of int64 type
#     if query[col].dtypes in ['float64', 'int64']:
#         scaler.fit(query[[col]])
#         tracks_scaled[col] = scaler.transform(query[col].values.reshape(-1,1)).ravel()
# print("Done Normalizing")

#km = KMeans()
# sse = [] # sse value for each k
# for i in range(1,75):
#     km = KMeans(n_clusters = i)
#     km.fit(tracks_scaled.sample(1000))
#     # calculating sse
#     sse.append(km.inertia_)

## Checking k values

#plt.plot(k_rng,sse)
#plt.xlabel('K value')
#plt.ylabel('SSE Error')
#plt.title('Best K value')
# plt.ylim(0,400)
# plt.xlim(0,100)
#plt.show()


## Predicting Genre
# km = KMeans(n_clusters=25)
# predicted_genres = km.fit_predict(model2[features_model2])
# model2['cluster_assignment'] = predicted_genres
# km = faiss.Kmeans(d=tracks[features].shape[1], k=1000, niter=300, nredo=10)
# numpyarr = tracks[features].to_numpy()
# numpyarr = numpyarr.copy(order='C')
# km.train(numpyarr)
# tracks_scaled["popularity_bin"] = query["popularity_bin"]
# tracks_scaled["id"] = query["id"]
# tracks_scaled["name"] = query["name"]
# tracks_scaled["artists"] = query["artists"]
# D, I = km.index.search(tracks[features], 1)
# model2.to_csv("Clustered.csv", index=False)
# print("Done training")
# tracks['cluster_assignment'] = predicted_genres


## Feature selection
# clf = LassoCV().fit(train_x, train_y)
# importance = np.abs(clf.coef_)
# idx_third = importance.argsort()[-3]
# threshold = importance[idx_third] + 0.01
# idx_features = (-importance).argsort()[:10]
# name_features = np.array(features)[idx_features]
# print('Selected features: {}'.format(name_features))

## Finding K value
# error_rate = []
# # Will take some time
# for i in range(1, 40):
#     knn = KNeighborsClassifier(n_neighbors=i)
#     knn.fit(train_x, train_y)
#     pred_i = knn.predict(train_x)
#     error_rate.append(np.mean(pred_i != train_y))
#
# ## Plotting Error
# plt.figure(figsize=(10,6))
# plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
#  markerfacecolor='red', markersize=10)
# plt.title('Error Rate vs. K Value')
# plt.xlabel('K')
# plt.ylabel('Error Rate')
# plt.show()
#
# @jit
# def adding_to(t, df):
#     for i in range(len(df)):
#         temp = df[i]
#         t.add_item(i, temp)



# print('starting query')
# selected_track = model2.loc[model1['id'] == "7dt6x5M1jzdTEt8oCbisTK"]
# selected_track = selected_track["cluster_assignment"]
# list_tracks = model2[model2['cluster_assignment'] == selected_track.values[0]]
# list_tracks = list_tracks.reset_index(drop=True)
# tracks_query = list_tracks[features_model2]
# list_tracks.to_csv('ClusteredSongs.csv', index=False)
#
# f = len(features_model2)
# t = AnnoyIndex(f, 'angular')
# adding_to(t, tracks_query.to_numpy())
# print('done adding')
# t.build(20) # 20 trees
# t.save('test.ann')
#
# u = AnnoyIndex(f, 'angular')
# u.load('test.ann')
#
# search_num = list_tracks[list_tracks['id'].str.contains('7dt6x5M1jzdTEt8oCbisTK')].index.values.astype(int)[0] + 2
# print(search_num)
# similar_songs = u.get_nns_by_item(search_num, 10)
# similar_songs = list_tracks.iloc[similar_songs]
# similar_songs.to_csv('SimilarSongs.csv', index=False)

# to_save = model1[model1['name'].str.contains('Just The Way')]
# to_save = to_save[to_save['artists'].str.contains('Bruno')]
# to_save.to_csv('BrunoSongs.csv', index=False)
# tracks.to_csv('Updated.csv', index=False)