from pathlib import Path
import os 
import numpy as np
import pandas as pd
from selenium import webdriver  
from selenium.webdriver.common.by import By

class CVPR_papers_collecter:
    def __init__(self,year=2023 ,data_path=Path("./papers_info")):
        """
        Args:
            year int: year
            data_path pathlib.Path: path to save papers info
        """
        self.conference_url = f"https://openaccess.thecvf.com/CVPR{year}?day=all"
        self.data_path = data_path

        # ディレクトリがなければ作成
        if os.path.exists(self.data_path) == False:
            os.makedirs(self.data_path)
        
        if not (2020 < year <= 2023):
            raise NotImplementedError
    
    def 