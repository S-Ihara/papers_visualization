# papers_visualization
論文をt-sneやumapなどで可視化させる
![](./images/papers_visualize.png "可視化した結果図")

## 環境構築
基本はrequirements.txtを使って環境作成してください`pip install -r requirements.txt`  
一応pyproject.toml作ったけど動くか確認してないです  
## Usage
- step.1 papers_collect.pyで論文情報をダウンロードする
    ```bash
    python papers_collect.py --conf cvpr --year 2023
    ```
- step.2 visualize_notebookかpapers_visualize.pyで可視化させる
    ```bash
    python papers_visualize.py
    ```

## ダウンロードできる国際会議論文
- cvpr (2023-2021)  

そのうち増やす

## Refetence
- https://github.com/dkobak/iclr-tsne/tree/main
