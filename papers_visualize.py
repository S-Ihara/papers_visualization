import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KernelDensity
from sklearn.manifold import TSNE
import umap

def papers_load(data_path,conf="all",year="all"):
    """
    Args:
        data_path Union[str,Path]: path to the data
        conf str: conference name "all" or specific conference name
        year Union[int,str]: year "all" or specific year
    Returns:
        papers pd.DataFrame: papers information
    """
    data_path = Path(data_path)

    if conf == "all":
        papers_file = glob.glob(str(data_path / '*.pickle'))
        papers = pd.concat([pd.read_pickle(paper_file) for paper_file in papers_file])
        if "decision" in papers.columns:
            papers = papers[papers['decision'] != 'Reject']
    else:
        papers_file = glob.glob(str(data_path / f'{conf}_*.pickle'))
        papers = pd.concat([pd.read_pickle(paper_file) for paper_file in papers_file])
        if "decision" in papers.columns:
            papers = papers[papers['decision'] != 'Reject']

    if len(papers_file) == 0:
        raise ValueError(f"papers pickle is not found in {data_path}.")
    
    if year != "all":
        papers = papers[papers['year'] == year]
    
    return papers

def keyword_extract(papers):
    """タイトルからよく用いられているキーワードを抽出する
    Args:
        papers pd.DataFrame: papers information
    Returns:
        keywords list[str]: keywords
    """
    keywords = []
    words, counts = np.unique(' '.join(papers.title).lower().split(), return_counts=True)
    ind = np.argsort(counts)[::-1][:30]
    for i in ind:
        if len(words[i]) >= 5: 
            #print(f'{words[i]:20} {counts[i]:4}')
            keywords.append(words[i])
    
    # 基本的すぎるキーワードの削除
    if 'learning' in keywords:keywords.remove('learning')
    if 'training' in keywords:keywords.remove('training')
    if 'image' in keywords:keywords.remove('image')
    if 'neural' in keywords:keywords.remove('neural')
    if 'towards' in keywords:keywords.remove('towards')
    if 'models' in keywords:keywords.remove('models')
    if 'model' in keywords:keywords.remove('model')
    if 'network' in keywords:keywords.remove('network')
    if 'networks' in keywords:keywords.remove('networks')

    return keywords

def embedding(title_abstract, mode="tfidf", mapping="umap", **kwargs):
    """タイトルとアブストからembeddingを作成する
    Args:
        title_abstract list[str]: タイトルとアブストの結合のリスト
        mode str: embeddingのモード "tfidf"
        mapping str: embeddingを二次元にマッピングする方法 "umap" "tsne"
    Returns:
        embedding np.ndarray: embedding shape=(n_samples, 2)
    """
    if mode == "tfidf":
        vectorizer = TfidfVectorizer() # ここパラメータいじれる
        embedding = vectorizer.fit_transform(title_abstract)
    else:
        raise NotImplementedError
    
    # mappingのパラメータ
    # umap
    n_neighbors = kwargs.get("n_neighbors", 20)
    min_dist = kwargs.get("min_dist", 0.1)
    metric = kwargs.get("metric", 'cosine')
    # tsne
    learning_rate = kwargs.get("learning_rate", 'auto')
    init = kwargs.get("init", 'random')
    perplexity = kwargs.get("perplexity", 10)

    if mapping == "umap":
        embedding = umap.UMAP(n_neighbors=20, min_dist=0.1, metric='cosine', verbose=True).fit_transform(embedding)
    elif mapping == "tsne":
        embedding = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=10, verbose=True).fit_transform(embedding)
    else:
        raise NotImplementedError

    return embedding

def papers_visualize(papers, papers_embedding, keywords, save_path=None, title="papers_visualize"):
    """papersを可視化する
    Args:
        papers pd.DataFrame: papers information
        papers_embedding np.ndarray: papers embedding shape=(n_samples, 2)
        keywords list[str]: keywords
        save_path Union[str,Path]: 保存先
    """
    # visualize
    fig,ax = plt.subplots(figsize=(10,10))

    ax.scatter(papers_embedding[:,0],papers_embedding[:,1],s=2,c="k",ec="none")
    ax.set_xticks([])
    ax.set_yticks([])

    # color map
    n = len(keywords)
    color = plt.get_cmap("gist_rainbow",n) # "jet" "gist_rainbow" "gist_ncar"

    for num, keyword in enumerate(keywords):
        idx = [i for i,t in enumerate(papers.title) if keyword.lower() in t.lower()]
        #ax.scatter(papers_embed[idx,0], papers_embed[idx,1], s=1, c=labelColors[num])
        ax.scatter(papers_embedding[idx,0], papers_embedding[idx,1], s=1, c=color(num))

    for num, keyword in enumerate(keywords):
        idx = [keyword.lower() in t.lower() for t in papers.title] 
        kde = KernelDensity(kernel='gaussian', bandwidth="scott").fit(papers_embedding[idx]) # "silverman" "scott"
        log_density = kde.score_samples(papers_embedding[idx])
        mode = papers_embedding[idx][np.argmax(log_density)]
        mode += mode / np.linalg.norm(mode)
        ax.text(mode[0], mode[1], keyword, ha='center', va='center', c='k', fontsize=6,
                bbox=dict(facecolor='w', alpha=1, edgecolor=color(num), boxstyle='round, pad=.2', linewidth=.5))

    if save_path is None:
        plt.show()
        return 

    if isinstance(save_path, str):
        save_path = Path(save_path)
    plt.savefig(save_path / (title+".png"), bbox_inches='tight', pad_inches=0.1)


if __name__ == "__main__":
    data_path = Path("./papers_info")
    save_path = Path("./results")

    papers = papers_load(data_path, conf="all", year="all")
    keywords = keyword_extract(papers)
    print("keywords are")
    print(keywords)
    title_and_abstract = [paper.title + ' ' + paper.abstract for idx, paper in papers.iterrows()]

    papers_embedding = embedding(title_and_abstract, mode="tfidf", mapping="umap")
    papers_visualize(papers, papers_embedding, keywords, save_path)
