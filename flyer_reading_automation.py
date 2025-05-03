from google.cloud import vision
from google.oauth2 import service_account
import requests
from bs4 import BeautifulSoup
import time
import datetime
import pytz
import glob
import pandas as pd
import re
import random
import mojimoji
from PIL import Image, ImageDraw
import os
from pathlib import Path
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

# __file__でスクリプトのパスを取得
# os.path.abspath()で絶対パスに変更
# os.path.dirname()でパスからそのディレクトリを取得
# os.chdir()でそのディレクトリにカレントディレクトリを変更
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class LINENotifyBot(object):
    API_URL = 'https://notify-api.line.me/api/notify'
    def __init__(self, access_token):
        self.__headers = {'Authorization': 'Bearer ' + access_token}

    def send(
        self,
        message,
        image=None,
        sticker_package_id=None,
        sticker_id=None,
    ):
        payload = {
            'message': message,
            'stickerPackageId': sticker_package_id,
            'stickerId': sticker_id,
        }
        files = {}
        if image != None:
            files = {'imageFile': open(image, 'rb')}
        r = requests.post(
            LINENotifyBot.API_URL,
            headers=self.__headers,
            data=payload,
            files=files,
        )

Yobikomi = LINENotifyBot(access_token=APIKEY)

def humanLikeRequestGet(url, std_interval=5, max_fluctuation=0, user_agent_rotation=True):
    if (std_interval - max_fluctuation) < 1:
        raise ValueError("humanLikeRequestGet: 現在の引数では、待機時間が1秒未満になる恐れがあります。std_interval=%dの場合、max_fluctuationは%d秒以下に設定してください。" %(std_interval, std_interval-1))
    elif max_fluctuation < 0:
        raise ValueError("humanLikeRequestGet: max_fluctuationには、0以上の整数を指定してください。")
    
    user_agents = [
        "Mozilla/5.0 (Windows NT 11.2; Win64; x64) AppleWebKit/610.2.9 (KHTML, like Gecko) Version/22.3 Edge/122.0.100.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/609.5.4 (KHTML, like Gecko) Vivaldi/6.8.3212.100 Chrome/126.0.6420.0 Safari/609.5.4"]
    
    if user_agent_rotation:
        designated_user_agent = random.choice(user_agents)
    else:
        designated_user_agent = user_agents[0]
    headers = {'User-Agent': designated_user_agent}
    
    response = requests.get(url, headers=headers)
    # requests.get()には、必ずtime.sleepを添えること。
    wait = std_interval + random.randint(-int(max_fluctuation), int(max_fluctuation))
    # print("%d秒待機します。" %wait)
    time.sleep(wait)

    return response

# URLで指定された店舗のチラシを取得する関数
def FlyersFromDesignatedShop(urls):
    # 新しくチラシが入った店舗名
    new_flyer_shop = set()
    # 新しいチラシの画像パスのリスト
    new_flyer_paths = []

    for url in urls:
        # htmlの取得はrequestsで行う。
        # User-Agentを指定（偽装）していないとアクセスできないよ。
        response = humanLikeRequestGet(url=url, std_interval=5, max_fluctuation=0, user_agent_rotation=True)
        # 自動検出されたエンコーディングを使用して文字化けを防止。
        response.encoding = response.apparent_encoding

        bs = BeautifulSoup(response.text, 'html.parser')
        # 店舗名を抽出。
        shop_name = bs.find(class_ = 'shop_name').text

        # 当該店舗のチラシ部分だけを抽出。
        leaflet = bs.find(id = 'leaflet')
        # チラシの掲載が無い時の例外処理。
        if leaflet is None:
            # ログの記録。
            df_log = pd.read_csv("./log_gets.csv", dtype=str)
            dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
            list = [dt_now.isoformat(timespec="milliseconds"),
                    -1,
                    shop_name,
                    'Hallucination.jpg',
                    'チラシ掲載なし']
            # 行として追加したい行をリストして、さらにネスト（入れ子）してから与えてね。
            df_add = pd.DataFrame(data=[list], columns=df_log.columns)
            # df_logに、試行のデータを新たな行として追加。
            # ignore_index=True は、新たなindexを割り振るためのオプション。
            df_log = pd.concat([df_log, df_add], axis=0, ignore_index=True)
            # log_gets.csvに、df_logを書き込む。
            df_log.to_csv("./log_gets.csv", index=False)

            # print(shop_name + "のページに、今日のチラシは見つかりませんでした。")
            continue
        # それぞれのチラシの領域に分けて考える。
        flyers = leaflet.find_all(class_ = 'image_element scroll track_hakari_once')
        # それぞれの領域から、チラシのURLと説明（更新日など）を抽出してリストに格納。
        img_urls = []
        descriptions = []
        for a_flyer in flyers:
            # アイコン画像などを回避しつつ、imgタグを含む要素を抽出。
            img_tag = a_flyer.find('img', class_ = 'image scroll lazy')
            # imgタグのdata-src属性から画像のURLを抽出。
            # 画像のURLをオリジナル画像のものへ書き換える。
            list = img_tag['data-src'].split('/')
            list[-2] = 'o=true'
            img_urls.append('/'.join(list))

            # 説明欄の抽出とリストへの追加。
            sentence = a_flyer.find(class_ = 'description').text
            descriptions.append(sentence.replace('\n', ''))

        # 重複取得防止のため、店舗名のフォルダ直下のファイル名を取得する。
        file_names = glob.glob("./img/" + shop_name + "/*")
        for i, file_name in enumerate(file_names):
            file_names[i] = file_name.split('/')[-1]

        # ここから、画像取得の試行とログの記録。
        df_log = pd.read_csv("./log_gets.csv", dtype=str)

        for i, img_url in enumerate(img_urls):
            # 当該画像が既に取得済みならリクエストしない。
            img_regular = re.search(r"/([^/]+\.jpg)\?", img_url).group(1)
            if img_regular in file_names:
                # ログの記録。
                dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
                list = [dt_now.isoformat(timespec="milliseconds"),
                        0,
                        shop_name,
                        img_regular,
                        descriptions[i]]
                # 行として追加したい行をリストして、さらにネスト（入れ子）してから与えてね。
                df_add = pd.DataFrame(data=[list], columns=df_log.columns)

                # print(img_regular + " は、既に取得済みです。")
            else:
                # 新しくチラシが入った店舗名を集合へ追記
                new_flyer_shop.add(shop_name)

                # ログの記録。
                dt_now = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
                list = [dt_now.isoformat(timespec="milliseconds"),
                        1,
                        shop_name,
                        img_regular,
                        descriptions[i]]
                # 行として追加したい行をリストして、さらにネスト（入れ子）してから与えてね。
                df_add = pd.DataFrame(data=[list], columns=df_log.columns)

                response = humanLikeRequestGet(url=img_url, std_interval=5, max_fluctuation=0, user_agent_rotation=True)
                # imageには、画像のバイナリデータが格納される。
                image = response.content

                # 画像の保存パス
                file_name = "./img/" + shop_name + "/" + img_regular
                # 新しいチラシの画像パスを追記
                new_flyer_paths.append(file_name)

                # バイナリファイルを読み書きするから、引数modeは'wb'。
                with open(file_name, 'wb') as f:
                    f.write(image)

            # df_logに、試行のデータを新たな行として追加。
            # ignore_index=True は、新たなindexを割り振るためのオプション。
            df_log = pd.concat([df_log, df_add], axis=0, ignore_index=True)

        # log_gets.csvに、df_logを書き込む。
        df_log.to_csv("./log_gets.csv", index=False)

    # LINE Notify
    if len(new_flyer_shop) == 0:
        # message = "スーパーのチラシを確認しましたが、新しいものはありませんでした。また確認し、連絡させていただきます。"
        # print(message)
        # Yobikomi.send(message=message)
        pass
    else:
        # shops = '\n・'.join(new_flyer_shop) # ここ list(new_flyer_shop) で渡せないんだけどなんで？？？
        # message = "スーパーのチラシを確認したところ、\n・" + shops + "\nの新しいチラシが出ていました。あらかじめ設定されているキーワードが含まれている場合は、書き込みを加えたチラシを以下に通知致します。"
        # print(message)
        # Yobikomi.send(message=message)
        pass
    
    return new_flyer_paths

def isOverlap(rect1, rect2):
    # 四角形の左上と右下の座標
    # 1個目の四角形
    LU_x_1, LU_y_1 = rect1[:2]
    RD_x_1, RD_y_1 = rect1[2:4]
    # 2個目の四角形
    LU_x_2, LU_y_2 = rect2[:2]
    RD_x_2, RD_y_2 = rect2[2:4]

    # 「重なり合ってない条件」の「否定」を返す。要は、重なり合っていたら Trueである。
    return not (RD_x_1<LU_x_2 or LU_x_1>RD_x_2 or RD_y_1<LU_y_2 or LU_y_1>RD_y_2)

def merge_overlapped_rects(rects):
    merged_rects = []
    checked = [False] * len(rects)

    for i in range(len(rects)):
        if checked[i]:
            continue
        merged_rect = rects[i]
        for j in range(i+1, len(rects)):
            if checked[j] or not isOverlap(merged_rect, rects[j]):
                continue
            # 重なっている場合、merged_rectを更新
            merged_rect = [
                min(merged_rect[0], rects[j][0]),
                min(merged_rect[1], rects[j][1]),
                max(merged_rect[2], rects[j][2]),
                max(merged_rect[3], rects[j][3])
            ]
            checked[j] = True
        merged_rects.append(merged_rect)
        checked[i] = True

    return merged_rects

# 指定されたフォルダのサイズをバイト単位で取得する関数
def get_folder_size(path):
    path = Path(path)
    return sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())

#######################################################################################################################

# チラシをクローリング

url_list = ["https://fake.fake.fake/バーチャルスーパーマーケット東京/fake"]

new_flyer_paths = FlyersFromDesignatedShop(urls=url_list)

#######################################################################################################################

for new_flyer_path in new_flyer_paths:
    # ここから、Cloud Vision APIに画像を送信して、その画像のfull_text_annotationを取得する作業に入る。

    # 身元証明書のjson読み込み
    credentials = service_account.Credentials.from_service_account_file("./APIKEY.json")

    client = vision.ImageAnnotatorClient(credentials=credentials)

    image_file = new_flyer_path
    with open(image_file, 'rb') as fb:
        content = fb.read()

    image = vision.Image(content=content)

    response = client.document_text_detection(image=image)
    document = response.full_text_annotation

    # ここから、パラグラフをcsvに書き込む作業に入る。
    # まずは、列ごとにリストでまとめる。次に、行ごとのリストに成形する。

    # パラグラフを格納するリスト
    list_paragraphs = []
    # パラグラフのバウンドの4頂点の座標を格納するリスト
    # vertices_0_x, vertices_0_y, vertices_1_x, vertices_1_y, vertices_2_x, vertices_2_y, vertices_3_x, vertices_3_y
    list_paragraphs_vertices = [[], [], [], [], [], [], [], []]

    # ページ単位で処理
    for page in document.pages:
        # ブロック単位で処理
        for block in page.blocks:
            # パラグラフ単位で処理
            for paragraph in block.paragraphs:
                # ワードの親パラグラフをあらかじめ保存しておく。
                # 二重の内包表記
                paragraph_text = ''.join([symbol.text for word in paragraph.words for symbol in word.symbols])
                list_paragraphs.append(paragraph_text)

                # シンボルの4頂点の座標の格納
                for i in range(4):
                    list_paragraphs_vertices[2*i].append(paragraph.bounding_box.vertices[i].x)
                    list_paragraphs_vertices[(2*i)+1].append(paragraph.bounding_box.vertices[i].y)

    # 列ごとのリストを、行ごとのリストに成形していく。
    # 行のリストが、data_main にネスト（入れ子）される。
    data_main = []
    for i in range(len(list_paragraphs)):
        record = []
        record.append(list_paragraphs[i])
        for j in range(8):
            record.append(list_paragraphs_vertices[j][i])

        # 1行分をデータフレーム用リストdataにネストする。
        data_main.append(record)

    # data_main をデータフレーム化する。
    # 行として追加したい行をリストして、さらにネスト（入れ子）してから与えてね。
    df_main = pd.DataFrame(data=data_main, columns=['paragraph', 'vertices_0_x', 'vertices_0_y', 'vertices_1_x', 'vertices_1_y', 'vertices_2_x', 'vertices_2_y', 'vertices_3_x', 'vertices_3_y'])

    # 英字・数字など（漢字・ひらがな・カタカナ以外全て）を半角に一括変換
    df_main['paragraph'] = df_main['paragraph'].apply(mojimoji.zen_to_han, kana=False)

    # csvへ書き込み
    df_main.to_csv("./paragraphs.csv", index=False)

    # 引っ掛けたいキーワードのリスト
    # このリストは、空っぽでも正しく動作するよ！
    keywords = ['キャノーラ', '上白糖', 'あさげ', 'ゆうげ', 'オリーブオイル', 'カップヌードル']
    # ヒットしたキーワードの集合
    hit_keywords = set()
    # ヒットしたインデックスの集合
    hit_index = set()
    for keyword in keywords:
        for i in range(len(df_main)):
            if keyword in df_main['paragraph'][i]:
                hit_keywords.add(keyword)
                hit_index.add(i)

    # ヒットしたインデックスの、データフレームに対応したブール値のリストを作る。（抽出のため。）
    hit_index_bool = []
    for i in range(len(df_main)):
        if i in hit_index:
            hit_index_bool.append(True)
        else:
            hit_index_bool.append(False)

    # ヒットした行だけをデータフレームに残す。
    df_main = df_main[hit_index_bool]

    # 発見場所をチラシに書き込んでいく。
    image = Image.open(image_file)
    draw = ImageDraw.Draw(image)

    # バウンドのまま描くと文字と線がかぶるため、広げる。
    expand = 65
    df_main['vertices_0_x'] = df_main['vertices_0_x'] - expand
    df_main['vertices_0_y'] = df_main['vertices_0_y'] - expand
    df_main['vertices_2_x'] = df_main['vertices_2_x'] + expand
    df_main['vertices_2_y'] = df_main['vertices_2_y'] + expand

    # バウンドの領域が被っていたら一つのバウンドにまとめていこう。
    list_LU_x = df_main['vertices_0_x'].tolist()
    list_LU_y = df_main['vertices_0_y'].tolist()
    list_RD_x = df_main['vertices_2_x'].tolist()
    list_RD_y = df_main['vertices_2_y'].tolist()

    # rect = [左上_x, 左上_y, 右下_x, 右下_y]
    # rects は、rect をネストしたもの。
    rects = []
    for LU_x, LU_y, RD_x, RD_y in zip(list_LU_x, list_LU_y, list_RD_x, list_RD_y):
        rect = [LU_x, LU_y, RD_x, RD_y]
        rects.append(rect)

    # ここで領域を統合
    merged_rects = merge_overlapped_rects(rects=rects)
    # 統合アルゴリズムに穴があるみたい。とりあえず2重でかけて対処する。
    merged_rects = merge_overlapped_rects(rects=merged_rects)

    # チラシ画像にキーワード検出位置の書き込みを行う。
    for i in range(len(merged_rects)):
        for j in range(4):
            draw.rounded_rectangle(
                [(merged_rects[i][0], merged_rects[i][1]), (merged_rects[i][2], merged_rects[i][3])],
                radius=expand,
                outline='blue',
                width=16
                )
            draw.rounded_rectangle(
                [(merged_rects[i][0], merged_rects[i][1]), (merged_rects[i][2], merged_rects[i][3])],
                radius=expand,
                outline='white',
                width=4
                )

    # 画像の保存
    match = re.search(r"(.+\d+)(\..+)", image_file)
    image_file_new = match.group(1) + "_Scanned" + match.group(2)
    image.save(image_file_new)

    # ここから、LINE Notify での送信を行う。
    # 今は、キーワードに引っかかった時だけ通知するように設定している。
    if len(hit_keywords) >= 1:
        # log_gets.csv から'説明欄'と'店舗名'を持ってくる。
        df_log = pd.read_csv("./log_gets.csv", dtype=str)
        match = re.search(r".+/(\d+\..+)", image_file)
        df_log = df_log[df_log['画像名'] == match.group(1)].iloc[-1]

        match = re.match(r"(.+) (.+)", df_log['店舗名'])
        shop_name_left = match.group(1)

        for url in url_list:
            if shop_name_left in url:
                url_shop = url
                break
            
        # message = '【' + df_log['店舗名'] + "】（" + df_log['説明欄'] + "）\nチラシ掲載キーワード：" + ', '.join(hit_keywords) + '\n元のチラシを見る：' + url_shop
        message = '【' + df_log['店舗名'] + "】（" + df_log['説明欄'] + "）\nチラシ掲載キーワード：" + ', '.join(hit_keywords)
        print(message)
        Yobikomi.send(message=message, image=image_file_new)

#######################################################################################################################

# imgファイルのサイズを予測する機能
# 結構、画像ファイルたちが大きいのです。

# ここから、UNIXタイムスタンプと、その時点でのimgフォルダのサイズ[B]をcsvに記録していく。
df_size = pd.read_csv("./img_folder_size.csv")
list = [int(time.time()), get_folder_size("./img")]
df_add = pd.DataFrame(data=[list], columns=df_size.columns)
df_size = pd.concat([df_size, df_add], axis=0, ignore_index=True)
df_size.to_csv("./img_folder_size.csv", index=False)

# ここから、線形回帰予測を行なっていく。

# 説明変数（Numpy配列）（一応、2次元配列という体裁をつける必要があるらしい。）
# x = df_size[['UNIXタイムスタンプ']].values
# 目的変数（Numpy配列）
# y = df_size['imgフォルダのサイズ'].values

# model = LinearRegression()
# model.fit(x, y)

# print('係数:', model.coef_[0])
# print('切片:', model.intercept_)

# plt.scatter(x, y, color='blue')
# # 回帰直線をプロット
# plt.plot(x, model.predict(x), color='red')
# plt.xlabel('UNIX timestamp')
# plt.ylabel('Size of img folder [B]')
# plt.grid()
# plt.show()

# 1ヶ月後のUNIXタイムスタンプを計算
# new_unix = [[int(time.time()) + 2592000]]
# GB単位で未来のサイズを予測
# size_pred = model.predict(new_unix)[0] / (10 ** 9)
# size_pred = np.round(size_pred, decimals=3)
# print("1ヶ月後のimgフォルダのサイズ予測:", size_pred, 'GB')

#######################################################################################################################

print('*' * 25)
