from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from selenium import webdriver  
from selenium.webdriver.common.by import By

class CVPR_papers_collecter:
    """Collect papers info from CVPR
    Note:
        Only 2021 ~ 2023 are supported
    """
    def __init__(self,year=2023 ,data_path=Path("./papers_info")):
        """
        Args:
            year int: year
            data_path pathlib.Path: path to save papers info
            chromedriver_path str: path to chromedriver
        """
        self.conference_url = f"https://openaccess.thecvf.com/CVPR{year}?day=all"
        self.data_path = data_path
        self.titles = []
        self.abstracts = []

        # ディレクトリがなければ作成
        if os.path.exists(self.data_path) == False:
            os.makedirs(self.data_path)
        
        self.year = year
        if not (2020 < year <= 2023):
            raise NotImplementedError
        
        #chromedriver_path = '/usr/bin/chromedriver'
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument("--no-sandbox")

        self.driver = webdriver.Chrome(
            options=options
        )
        self.driver.get(self.conference_url)
    
    def collect(self):
        """Collect papers info
        """
        if 2020 < self.year <= 2023:
            title_element_list = self.driver.find_elements(By.CLASS_NAME, 'ptitle')

            paper_links = []
            for element in tqdm(title_element_list):
                paper_links.append(element.find_element(By.TAG_NAME, 'a').get_attribute('href'))
            print(f"{len(paper_links)} papers are found")

            for i,element in enumerate(tqdm(title_element_list)):
                self.driver.get(paper_links[i])
                # if Not Found page is shown, skip
                if self.driver.find_element(By.TAG_NAME, 'body').text[:9] == 'Not Found':
                    continue

                title = self.driver.find_element(By.ID, 'papertitle').text
                abst = self.driver.find_element(By.ID, 'abstract').text
                self.titles.append(title)
                self.abstracts.append(abst)
            print(f"{len(self.titles)} papers are collected")

        else:
            raise NotImplementedError
    
    def save_pickles(self,mask_length=200,save_path=None):
        """Save papers info as pickle
        Args:
            mask_length int: if abstract length is below this value, remove the paper
            save_path Union[pathlib.Path,str]: path to save papers info
        """
        cvpr = pd.DataFrame.from_dict({
            'year': self.year,
            'title': self.titles, 
            'abstract': self.abstracts,
            'conference': "cvpr",
        })

        mask = np.array([len(a) >= 200 for a in cvpr.abstract])
        cvpr = cvpr[mask].reset_index(drop=True)
        print(f'Removing {np.sum(~mask)} submissions with abstract length below 200 characters.')

        if isinstance(save_path, str):
            save_path = Path(save_path)
        cvpr.to_pickle(save_path / f'cvpr_{self.year}.pickle')
        print(f"cvpr_{self.year}.pickle is saved at {save_path}")