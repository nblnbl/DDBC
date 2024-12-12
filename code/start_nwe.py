from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.decomposition import PCA
from utils import *
from collections import Counter
from ahc import *
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from Eu import *
import math

def start_new(train_x,train_y,query_x,query_y,p,lamda = 0.7,k = 3):
    X = np.concatenate((train_x, query_x), axis=0)
    Y = np.concatenate((train_y, query_y), axis=0)
    iris = pd.read_csv("data/wine.csv", header=0)
    df = iris 
    # print(df)
    columns = list(df.columns)  
    features = columns[:len(columns) - 1]  
    dataset = df[features]  

    # X = np.array(dataset).astype(np.float32)
    # attributes = len(df.columns) - 1 
    # original_labels = list(df[columns[-1]])  
    # if type(original_labels[0]) != int:
    #     Y = LabelEncoder().fit_transform(df[columns[len(columns) - 1]])
    # else:
    #     Y = original_labels

    scaler = StandardScaler()
    X = scaler.fit_transform(X)


    distances, index_matrix, point_categories, is_core,delts = compute_and_identify_points(X,k,p)
    knn_distances = compute_knn_distances(index_matrix, distances)
    clusters = [set() for _ in range(1000)]  
    clusters_num = 0 
    clusters_label = np.full(1000, -1, dtype=int) 
    is_cluster = np.zeros(len(X), dtype=bool) 
    x_cluster = np.full(len(X), -1, dtype=int)  
    x_far = np.full(len(X), -1, dtype=int)
    is_c = set()
    tatol_clusters = set()
    #1.构建初始簇
    for i in range(len(X)):
        if point_categories[i] == 1 and is_cluster[i] == False:
            clusters[clusters_num].add(i) 
            x_cluster[i] = clusters_num
            tatol_clusters.add(clusters_num)
            clusters_num = clusters_num + 1
            is_cluster[i] = True
            x_far[i] = i

            flag = True
            new_num = x_cluster[i]
            clusters_copy = list(clusters[new_num])
            while flag:
                for j in clusters_copy:
                    if point_categories[j] == 1:
                        for a in index_matrix[j]:
                            # if j >= len(train_x) and a >= len(train_x):
                            #     continue
                            if a == j:
                                continue
                            if is_cluster[a] == False :
                                clusters[new_num].add(a)
                                x_cluster[a] = new_num
                                is_cluster[a] = True
                                x_far[a] = j
                            else:
                                if point_categories[a] == 1:
                                    if x_cluster[a] != x_cluster[j]:
                                        tatol_clusters.discard((x_cluster[a]))
                                        for x in clusters[x_cluster[a]]:
                                            clusters[x_cluster[j]].add(x)
                                            x_cluster[x] = x_cluster[j]
                                        x_far[a] = j#???????
                if clusters_copy == list(clusters[new_num]):
                    flag = False
                else:
                    clusters_copy = list(clusters[new_num])
                # print(clusters)
    # clusters, clusters_label, clusters_num, x_cluster = get_clusters(X, clusters, clusters_label, clusters_num,
    #                                                                  tatol_clusters)
    # tatol_clusters = set()
    # for i in range(clusters_num):
    #     tatol_clusters.add(i)
    clusters_num = len(tatol_clusters)
    clusters_new = [set() for _ in range(1000)]
    a = 0
    for i in tatol_clusters:
        clusters_new[a] = clusters[i]
        a += 1
    tatol_clusters = set()
    clusters_label = np.full(1000, -1, dtype=int)
    for i in range(clusters_num):
        tatol_clusters.add(i)
    x_cluster = np.full(len(X), -1, dtype=int)
    l = 0
    pred = np.full(len(X), -1, dtype=int)
    for i in tatol_clusters:
        for j in clusters_new[i]:
            pred[j] = l
            x_cluster[j] = i
        clusters_label[i] = l
        l += 1
    clusters_x = []
    for i in range(clusters_num):
        clusters_x.append(compute_proto(clusters[i],X))

    a = 0

    for i in range(len(X)):
        if pred[i] == -1:
            confidences_p = compute_confidence(X[i], clusters_x, clusters_label,tatol_clusters)
            knn_confidences = compute_knn_confidence(X[i], X, pred,delts[i],k = 2)
            confidences = [lamda * cp + (1 - lamda) * kn for cp, kn in zip(confidences_p, knn_confidences)]
            pred[i] = np.argmax(confidences)
            # clusters_new[np.argmax(confidences)-1].add(i)
            # distances = np.full(clusters_num, -1, dtype=int)
            # for j in range(clusters_num):
            #     distances[j] = euclidean_distance(X[i],x_cluster[j])
            # clusters_new[np.argmin(distances)].add(i)
            # pred[i] = clusters_label[np.argmin(distances)]
            # a += 1
    # print(a)
    pred = np.sort(pred)
    Y = np.sort(Y)
    # print(pred)
    # print(Y)
    # print(clusters_new)
    # print(tatol_clusters)
    # for i in clusters_new[1]:
    #     print(index_matrix[i])
    acc = calculate_accuracy(pred, Y)
    nmi = normalized_mutual_info_score(pred, Y)
    ari = adjusted_rand_score(pred, Y)

    #计算轮廓系数
    sh = silhouette_score(X,clusters_new,x_cluster,clusters_num) / clusters_num
    # sh = inter_cluster_distance(X, clusters, clusters_num)
    # sh = calculate_separation(X, clusters, clusters_num)

    # print(acc,nmi,ari,sh)

    plt.cla()
    K = len(set(Y))
    colors = np.array(["blue", "red", "green", "black", "purple", "cyan", "orange", "hotpink", "hotpink", "#88c999", "#88c999", "#88c999"])
    if len(X[0]) > 2:
        datas = PCA(n_components=2).fit_transform(X)  
    label = pred
    for i in range(0,8):
        plt.scatter(datas[np.nonzero(label == i), 0], datas[np.nonzero(label == i), 1], c=colors[i], s=7)
        # plt.scatter(datas[centers[i], 0], datas[centers[i], 1], color="k", marker="+", s=200.) 
    # plt.savefig(file_name + "_cluster.jpg")

    # plt.xticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.yticks(fontproperties='Times New Roman', fontsize=10.5)
    # plt.show()
    clusters = clusters_new
    # for i in clusters[0]:
    #     print(Y[i])
    # print(is_core[:3])
    return acc, nmi ,ari,sh
