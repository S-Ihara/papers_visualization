from pathlib import Path
import os
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd

class ICLR_papers_collecter:
    """Collect papers info from ICLR
    """
    def __init__(self,year=2023 ,data_path=Path("./papers_info")):
        """
        Args:
            year int: year
            data_path pathlib.Path: path to save papers info
            chromedriver_path str: path to chromedriver
        """
        self.data_path = data_path
        self.year = year
        self.titles = []
        self.abstracts = []
        self.df = None

        self.url = f'https://api.openreview.net/notes?invitation=ICLR.cc%2F{year}%2FConference%2F-%2FBlind_Submission'

    def collect(self):
        offset = 0
        for i in range(100):
            offset = i * 1000
            current_df = pd.DataFrame(requests.get(self.url + f'&offset={offset}').json()['notes'])
            self.df = pd.concat([self.df, current_df])
            if current_df.shape[0] == 0:
                break
        print(f"{self.df.shape[0]} papers are found")
        
        for i, row in tqdm(self.df.iterrows()):
            forum_id = row.forum
            forum_url = f'https://api.openreview.net/notes?forum={forum_id}'
            json = requests.get(forum_url).json()
            for i in range(len(json['notes'])):
                if 'decision' in json['notes'][i]['content']:
                    decision = json['notes'][i]['content']['decision']
            if "Accept" in decision:
                self.titles.append(row.content["title"])
                self.abstracts.append(row.content["abstract"])
        print(f"{len(self.titles)} papers are collected")

    def save_pickles(self,save_path,mask_length=200):
        iclr = pd.DataFrame.from_dict({
            'year': self.year,
            'title': self.titles, 
            'abstract': self.abstracts,
            'conference': "iclr",
        })
        assert len(self.titles) != 0, "self.titles is empty. Please run collect() first."

        mask = np.array([len(a) >= 200 for a in iclr.abstract])
        iclr = iclr[mask].reset_index(drop=True)
        print(f'Removing {np.sum(~mask)} submissions with abstract length below 200 characters.')

        if isinstance(save_path, str):
            save_path = Path(save_path)
        iclr.to_pickle(save_path / f'iclr_{self.year}.pickle')
        print(f"iclr_{self.year}.pickle is saved at {save_path}")